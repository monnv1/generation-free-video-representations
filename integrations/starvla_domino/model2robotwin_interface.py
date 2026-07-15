"""StarVLA ↔ DOMINO / RoboTwin evaluation interface.

This module implements the ``get_model`` / ``reset_model`` / ``eval`` API
expected by ``DOMINO/script/eval_policy.py`` (via ``eval_function_decorator``).

History support
---------------
By default the client sends only the **current frame** to the policy server.
Set ``history_k > 0`` (via ``deploy_policy.yml`` or ``get_model`` kwargs) to
enable an extensible *historical-context* pipeline:

* **optical-flow** (``history_mode="flow"``, default when history is on):
  Compute Farneback optical-flow RGB images between consecutive historical
  frames and pass them as ``example["history_images"]``.

* **raw-frames** (``history_mode="frames"``):
  Pass the last ``history_k`` raw RGB frames directly.

* **video-frame sequence** (``history_as_video_frames=true``):
  Replace ``example["image"]`` with video frames ``[t-k, ..., t-1, t]``.  If
  ``multi_view_concat=true``, each temporal frame is a horizontal concat of
  ``[head | left_wrist | right_wrist]`` so WM4A still receives a single video
  stream.

* **custom**: Subclass ``ModelClient`` and override ``_build_history_context``
  to implement any other historical representation.

The server-side model (``QwenOFT.predict_action``) currently ignores
``history_images`` — but the WebSocket transport passes it through
transparently.  When a model that consumes history is available (e.g.
PUMA), no client-side change is needed.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


def _load_history_utils():
    """Import history helpers without forcing a single import style."""
    try:
        from history_flow_utils import (
            compute_flow_rgb_farneback,
            parse_hw_size,
            sample_history_offsets,
        )
    except ImportError:
        from examples.DOMINO.eval_files.history_flow_utils import (
            compute_flow_rgb_farneback,
            parse_hw_size,
            sample_history_offsets,
        )
    return compute_flow_rgb_farneback, parse_hw_size, sample_history_offsets


# ---------------------------------------------------------------------------
# ModelClient
# ---------------------------------------------------------------------------

MODEL_TO_ENV_ACTION_ORDER = np.array([0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13])
ENV_TO_MODEL_ACTION_ORDER = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13])
CONTINUOUS_ACTION_MASK = np.array(
    [True, True, True, True, True, True, True, True, True, True, True, True, False, False],
    dtype=bool,
)

def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class _AsyncInferenceWorker:
    """Single-request background worker with its own WebSocket connection."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._requests: queue.Queue = queue.Queue(maxsize=1)
        self._results: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pending = False
        self._client: Optional[WebsocketClientPolicy] = None
        self._thread = threading.Thread(target=self._run, name="domino-async-inference", daemon=True)
        self._thread.start()

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending

    def submit(self, request: dict) -> bool:
        with self._lock:
            if self._pending or self._stop_event.is_set():
                return False
            self._pending = True
        try:
            self._requests.put_nowait(request)
            return True
        except queue.Full:
            with self._lock:
                self._pending = False
            return False

    def pop_latest_result(self) -> Optional[dict]:
        latest = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                return latest

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        if self._client is not None:
            self._client.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _ensure_client(self) -> WebsocketClientPolicy:
        if self._client is None:
            self._client = WebsocketClientPolicy(self._host, self._port)
        return self._client

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break

            result = {
                "ok": False,
                "request_id": request.get("request_id"),
                "step": request.get("step"),
                "epoch": request.get("epoch"),
                "submitted_at_s": request.get("submitted_at_s"),
                "state": request.get("state"),
                "prev_action": request.get("prev_action"),
            }
            try:
                websocket_start = time.perf_counter()
                response = self._ensure_client().predict_action(request["query"])
                result.update(
                    {
                        "ok": True,
                        "response": response,
                        "websocket_roundtrip_s": time.perf_counter() - websocket_start,
                        "finished_at_s": time.perf_counter(),
                    }
                )
            except Exception as exc:
                result.update({"error": repr(exc), "finished_at_s": time.perf_counter()})
                try:
                    if self._client is not None:
                        self._client.close()
                finally:
                    self._client = None
            finally:
                with self._lock:
                    self._pending = False
                self._results.put(result)


class ModelClient:
    """WebSocket client for StarVLA policy inference during DOMINO evaluation.

    Parameters
    ----------
    policy_ckpt_path : str
        Path to a StarVLA checkpoint directory.
    history_k : int
        Number of historical frames to include.  0 = current frame only.
    history_stride : int
        Temporal stride between sampled historical frames.
    history_mode : str
        ``"flow"`` → optical-flow RGB;  ``"frames"`` → raw RGB.
    history_as_video_frames : bool
        If true, send history plus current frame as ``example["image"]``.
        Use this for Cosmos Video2World backbones.
    multi_view_concat : bool
        If true, horizontally concatenate the three DOMINO camera views into
        one frame before building the history video sequence.
    history_image_size : tuple[int,int] | None
        Output resolution for history images.  Defaults to ``image_size``.
    history_flow_compute_size : tuple[int,int] | None
        Internal resolution for flow computation.  Defaults to ``(128, 128)``.
    """

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble: bool = False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] | None = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "127.0.0.1",
        port: int = 5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        # --- history / optical-flow ---
        history_k: int = 0,
        history_stride: int = 1,
        history_mode: str = "flow",
        history_as_video_frames: bool = False,
        multi_view_concat: bool = False,
        history_image_size: Optional[list[int]] = None,
        history_flow_compute_size: Optional[list[int]] = None,
        # --- async inference / chunk handoff ---
        async_inference: bool = False,
        chunk_transition: str = "replace",
        async_result_cache_size: int = 2,
        rtc_execution_horizon: int = 10,
        rtc_max_guidance_weight: float = 10.0,
        rtc_prefix_attention_schedule: str = "exp",
        rtc_inference_delay: Optional[Union[int, str]] = None,
        rtc_compensate_inference_delay: bool = True,
        rtc_blend_dims: Optional[Union[str, List[int]]] = None,
    ) -> None:
        if image_size is None:
            image_size = [224, 224]

        self.host = host
        self.port = port
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # --- History configuration ---
        self.history_k = max(0, int(history_k))
        self.history_stride = max(1, int(history_stride))
        self.history_mode = self._normalize_history_mode(history_mode)
        self.history_as_video_frames = _as_bool(history_as_video_frames)
        self.multi_view_concat = _as_bool(multi_view_concat)
        _, parse_hw_size, _ = _load_history_utils()
        self.history_enabled = self.history_k > 0 and self.history_mode != "none"
        self.history_image_size: Tuple[int, int] = parse_hw_size(history_image_size, default_size=tuple(image_size))
        self.history_flow_compute_size: Tuple[int, int] = parse_hw_size(
            history_flow_compute_size, default_size=(128, 128)
        )
        # Frame buffer for history — stores resized RGB arrays at flow_compute_size
        if self.history_enabled:
            buf_len = self.history_k * self.history_stride
            self.history_frame_buffer: deque[np.ndarray] = deque(maxlen=buf_len)
            print(
                f"*** history enabled: k={self.history_k}, stride={self.history_stride}, "
                f"mode={self.history_mode}, out_size={self.history_image_size}, "
                f"flow_compute_size={self.history_flow_compute_size}, "
                f"as_video_frames={self.history_as_video_frames}, "
                f"multi_view_concat={self.multi_view_concat} ***"
            )
        else:
            self.history_frame_buffer = deque(maxlen=0)

        self.action_norm_stats = self.get_action_stats(
            self.unnorm_key, policy_ckpt_path=policy_ckpt_path, action_mode=action_mode
        )
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.raw_actions = None
        self.last_timing: Dict[str, Optional[float]] = {}
        self._last_chunk_request_time: Optional[float] = None
        self._last_chunk_request_step: Optional[int] = None
        self._raw_action_cursor = 0
        self._active_chunk_source_step: Optional[int] = None
        self._active_chunk_request_id: Optional[str] = None
        self._request_seq = 0
        self._async_epoch = 0
        self.async_inference = _as_bool(async_inference)
        self.chunk_transition = self._normalize_chunk_transition(chunk_transition)
        self.async_result_cache = deque(maxlen=max(1, int(async_result_cache_size)))
        self.rtc_execution_horizon = max(0, int(rtc_execution_horizon))
        self.rtc_max_guidance_weight = max(0.0, float(rtc_max_guidance_weight))
        self.rtc_prefix_attention_schedule = self._normalize_rtc_schedule(rtc_prefix_attention_schedule)
        self.rtc_inference_delay = self._normalize_rtc_inference_delay(rtc_inference_delay)
        self.rtc_compensate_inference_delay = _as_bool(rtc_compensate_inference_delay)
        self.rtc_blend_dims = self._normalize_rtc_blend_dims(rtc_blend_dims)
        self.async_worker = _AsyncInferenceWorker(host, port) if self.async_inference else None
        print(
            f"*** async_inference: {self.async_inference}, "
            f"chunk_transition: {self.chunk_transition}, "
            f"async_result_cache_size: {self.async_result_cache.maxlen}, "
            f"rtc_execution_horizon: {self.rtc_execution_horizon}, "
            f"rtc_max_guidance_weight: {self.rtc_max_guidance_weight}, "
            f"rtc_prefix_attention_schedule: {self.rtc_prefix_attention_schedule}, "
            f"rtc_inference_delay: {self.rtc_inference_delay if self.rtc_inference_delay is not None else 'auto'}, "
            f"rtc_compensate_inference_delay: {self.rtc_compensate_inference_delay}, "
            f"rtc_blend_dims: {self.rtc_blend_dims if self.rtc_blend_dims is not None else 'auto'} ***"
        )

    # ---- lifecycle -------------------------------------------------------

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.history_frame_buffer.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self.initial_state = None
        self.prev_action = None
        self._last_chunk_request_time = None
        self._last_chunk_request_step = None
        self._raw_action_cursor = 0
        self._active_chunk_source_step = None
        self._active_chunk_request_id = None
        self.async_result_cache.clear()
        self._async_epoch += 1

    def close(self) -> None:
        self.client.close()
        if self.async_worker is not None:
            self.async_worker.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ---- history pipeline ------------------------------------------------

    @staticmethod
    def _normalize_history_mode(history_mode: Optional[str]) -> str:
        mode = "flow" if history_mode is None else str(history_mode).strip().lower()
        if mode in {"off", "none", "disabled"}:
            return "none"
        if mode not in {"flow", "frames"}:
            raise ValueError(
                f"Unknown history_mode: {history_mode!r}. Expected one of ['flow', 'frames', 'none']."
            )
        return mode

    def _push_history_frame(self, image: np.ndarray) -> None:
        """Push a frame into the history buffer."""
        if not self.history_enabled:
            return
        if self.history_mode == "flow":
            h, w = self.history_flow_compute_size
            image = cv.resize(image, (w, h), interpolation=cv.INTER_AREA)
        self.history_frame_buffer.append(np.asarray(image).copy())

    @staticmethod
    def _concat_views_horizontally(images: List[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("multi_view_concat requires at least one image.")
        return np.concatenate([np.asarray(image) for image in images], axis=1)

    def _build_model_video_frame(self, images: List[np.ndarray]) -> np.ndarray:
        if self.multi_view_concat:
            return self._concat_views_horizontally(images)
        return images[0]

    def _build_history_context(self, current_image: np.ndarray) -> Optional[List[np.ndarray]]:
        """Build history representation from the frame buffer.

        Override this method to implement custom history representations.

        Returns ``None`` when history is disabled or the buffer is empty.
        Otherwise returns a list of RGB arrays (one per historical slot).
        """
        if not self.history_enabled:
            return None

        if self.history_mode == "flow":
            return self._build_history_flow(current_image)
        if self.history_mode == "frames":
            return self._build_history_frames(current_image)
        return None

    def _build_history_payload(self, current_image: np.ndarray) -> Dict[str, object]:
        """Build extra model inputs derived from historical context.

        Override this method when a future history-aware model needs a
        different payload schema. The default keeps the transport backward
        compatible by sending `history_images` when history is enabled.
        """
        history_images = self._build_history_context(current_image)
        if history_images is None:
            return {}
        return {"history_images": history_images}

    def _build_history_video_frames(self, current_image: np.ndarray) -> Optional[List[np.ndarray]]:
        """Return historical video frames plus the current frame.

        Cosmos Video2World consumes a temporal sequence.  For that backend we
        put the sampled history directly in ``example["image"]``.  With
        ``multi_view_concat=true``, every temporal frame is one horizontally
        concatenated multi-camera image.
        """
        if not self.history_enabled or not self.history_as_video_frames:
            return None
        if self.history_mode != "frames":
            raise ValueError("history_as_video_frames requires history_mode='frames'.")

        out_h, out_w = current_image.shape[:2] if self.multi_view_concat else self.history_image_size
        frames = self._build_history_frames(current_image, output_size=(out_h, out_w))
        current = cv.resize(current_image, (out_w, out_h), interpolation=cv.INTER_AREA)
        frames.append(current)
        return frames

    def _build_history_flow(self, current_image: np.ndarray) -> List[np.ndarray]:
        """Compute optical-flow images between consecutive sampled history frames."""
        compute_flow_rgb_farneback, _, sample_history_offsets = _load_history_utils()

        h, w = self.history_flow_compute_size
        current_small = cv.resize(current_image, (w, h), interpolation=cv.INTER_AREA)
        offsets = sample_history_offsets(self.history_k, self.history_stride)

        buf = self.history_frame_buffer
        buf_len = len(buf)

        # Build list of sampled frames (clamped to buffer bounds)
        sampled: List[np.ndarray] = []
        for off in offsets:
            idx = buf_len + off  # off is negative
            idx = max(0, min(idx, buf_len - 1)) if buf_len > 0 else -1
            frame = buf[idx] if idx >= 0 else current_small
            frame = cv.resize(frame, (w, h), interpolation=cv.INTER_AREA)
            sampled.append(frame)

        # Add current frame at the end for the last flow pair
        sampled.append(current_small)

        # Compute flow between consecutive pairs → history_k flow images
        out_h, out_w = self.history_image_size
        flow_images: List[np.ndarray] = []
        for i in range(len(sampled) - 1):
            flow_rgb = compute_flow_rgb_farneback(
                sampled[i], sampled[i + 1], compute_size=self.history_flow_compute_size,
            )
            flow_rgb = cv.resize(flow_rgb, (out_w, out_h), interpolation=cv.INTER_AREA)
            flow_images.append(flow_rgb)

        return flow_images

    def _build_history_frames(
        self,
        current_image: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        """Return raw historical frames (resized to history_image_size)."""
        _, _, sample_history_offsets = _load_history_utils()

        offsets = sample_history_offsets(self.history_k, self.history_stride)

        buf = self.history_frame_buffer
        buf_len = len(buf)
        out_h, out_w = output_size if output_size is not None else self.history_image_size

        frames: List[np.ndarray] = []
        for off in offsets:
            idx = buf_len + off
            idx = max(0, min(idx, buf_len - 1)) if buf_len > 0 else -1
            frame = buf[idx] if idx >= 0 else current_image
            frame = cv.resize(frame, (out_w, out_h), interpolation=cv.INTER_AREA)
            frames.append(frame)

        return frames

    # ---- chunk handoff / inference helpers -------------------------------

    @staticmethod
    def _normalize_chunk_transition(chunk_transition: Optional[str]) -> str:
        mode = "replace" if chunk_transition is None else str(chunk_transition).strip().lower()
        aliases = {
            "blend": "rtc",
            "smooth": "rtc",
            "rtc_blend": "rtc",
            "rtc_smooth": "rtc",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"replace", "rtc"}:
            raise ValueError(
                f"Unknown chunk_transition: {chunk_transition!r}. "
                "Currently implemented: 'replace' and 'rtc'."
            )
        return mode

    @staticmethod
    def _normalize_rtc_schedule(schedule: Optional[str]) -> str:
        mode = "exp" if schedule is None else str(schedule).strip().lower()
        if mode not in {"linear", "exp", "ones", "zeros"}:
            raise ValueError(
                f"Unknown rtc_prefix_attention_schedule: {schedule!r}. "
                "Expected one of ['linear', 'exp', 'ones', 'zeros']."
            )
        return mode

    @staticmethod
    def _normalize_rtc_inference_delay(value: Optional[Union[int, str]]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"", "auto", "none", "null"}:
                return None
            value = int(text)
        delay = int(value)
        if delay < 0:
            raise ValueError(f"rtc_inference_delay must be non-negative or 'auto', got {value!r}.")
        return delay

    @staticmethod
    def _normalize_rtc_blend_dims(value: Optional[Union[str, List[int]]]) -> Optional[Union[str, List[int]]]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"", "auto", "mask", "continuous", "null"}:
                return None
            if text in {"all", "none"}:
                return text
            return [int(part.strip()) for part in text.split(",") if part.strip()]
        return [int(dim) for dim in value]

    def _resolve_rtc_inference_delay(
        self,
        source_step: Optional[int],
        activation_step: Optional[int],
    ) -> int:
        if self.rtc_inference_delay is not None:
            return self.rtc_inference_delay
        if source_step is None or activation_step is None:
            return 0
        return max(0, int(activation_step) - int(source_step))

    def _rtc_blend_mask(self, action_dim: int) -> np.ndarray:
        spec = self.rtc_blend_dims
        if spec == "all":
            return np.ones(action_dim, dtype=bool)
        if spec == "none":
            return np.zeros(action_dim, dtype=bool)

        if isinstance(spec, list):
            mask = np.zeros(action_dim, dtype=bool)
            for dim in spec:
                idx = int(dim)
                if idx < 0:
                    idx += action_dim
                if idx < 0 or idx >= action_dim:
                    raise ValueError(f"rtc_blend_dims contains invalid dim {dim} for action_dim={action_dim}.")
                mask[idx] = True
            return mask

        stats_mask = self.action_norm_stats.get("mask")
        if stats_mask is not None:
            stats_mask = np.asarray(stats_mask, dtype=bool).reshape(-1)
            if stats_mask.shape[0] == action_dim:
                return stats_mask.copy()
        return np.ones(action_dim, dtype=bool)

    def _rtc_prefix_weights(self, overlap_len: int, inference_delay: int) -> np.ndarray:
        if overlap_len <= 0 or self.rtc_max_guidance_weight <= 0:
            return np.zeros(max(0, overlap_len), dtype=np.float32)

        delay = max(0, min(int(inference_delay), overlap_len))
        if self.rtc_prefix_attention_schedule == "ones":
            schedule = np.ones(overlap_len, dtype=np.float32)
        elif self.rtc_prefix_attention_schedule == "zeros":
            schedule = np.zeros(overlap_len, dtype=np.float32)
            schedule[:delay] = 1.0
        else:
            schedule = np.ones(overlap_len, dtype=np.float32)
            tail_len = overlap_len - delay
            if tail_len > 0:
                if tail_len == 1:
                    tail = np.zeros(1, dtype=np.float32)
                else:
                    x = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)
                    if self.rtc_prefix_attention_schedule == "linear":
                        tail = 1.0 - x
                    else:
                        decay = 5.0
                        exp_tail = np.exp(-decay * x)
                        tail = (exp_tail - np.exp(-decay)) / (1.0 - np.exp(-decay))
                schedule[delay:] = tail

        strength = self.rtc_max_guidance_weight / (self.rtc_max_guidance_weight + 1.0)
        return np.clip(strength * schedule, 0.0, 1.0)

    @staticmethod
    def _empty_handoff_info() -> dict:
        return {
            "rtc_applied": False,
            "rtc_overlap_steps": 0,
            "rtc_inference_delay_steps": None,
            "rtc_new_start_index": 0,
            "rtc_blend_dim_count": 0,
            "rtc_old_weight_min": 0.0,
            "rtc_old_weight_max": 0.0,
        }

    def _build_rtc_handoff_actions(
        self,
        raw_actions: np.ndarray,
        source_step: Optional[int],
        activation_step: Optional[int],
    ) -> Tuple[np.ndarray, dict]:
        info = self._empty_handoff_info()
        inference_delay = self._resolve_rtc_inference_delay(source_step, activation_step)
        info["rtc_inference_delay_steps"] = inference_delay

        if len(raw_actions) == 0:
            return raw_actions, info

        new_start_index = 0
        if self.rtc_compensate_inference_delay:
            new_start_index = min(inference_delay, len(raw_actions) - 1)
        info["rtc_new_start_index"] = new_start_index
        candidate_actions = raw_actions[new_start_index:].copy()

        if (
            self.raw_actions is None
            or not self.async_inference
            or self._raw_action_cursor >= len(self.raw_actions)
            or self.rtc_execution_horizon <= 0
            or self.rtc_max_guidance_weight <= 0
        ):
            return candidate_actions, info

        old_leftover = self.raw_actions[self._raw_action_cursor:]
        overlap_len = min(self.rtc_execution_horizon, len(old_leftover), len(candidate_actions))
        if overlap_len <= 0:
            return candidate_actions, info

        blend_mask = self._rtc_blend_mask(candidate_actions.shape[-1])
        blend_dim_count = int(np.count_nonzero(blend_mask))
        info["rtc_overlap_steps"] = overlap_len
        info["rtc_blend_dim_count"] = blend_dim_count
        if blend_dim_count == 0:
            return candidate_actions, info

        old_weights = self._rtc_prefix_weights(overlap_len, inference_delay).astype(candidate_actions.dtype)
        if not np.any(old_weights > 0):
            return candidate_actions, info

        blended_actions = candidate_actions.copy()
        old_prefix = old_leftover[:overlap_len]
        new_prefix = candidate_actions[:overlap_len]
        blended_prefix = old_weights[:, None] * old_prefix + (1.0 - old_weights[:, None]) * new_prefix
        blended_actions[:overlap_len, blend_mask] = blended_prefix[:, blend_mask]

        info["rtc_applied"] = True
        positive_weights = old_weights[old_weights > 0]
        info["rtc_old_weight_min"] = float(np.min(positive_weights))
        info["rtc_old_weight_max"] = float(np.max(positive_weights))
        return blended_actions, info

    def _next_request_id(self, step: int, async_request: bool) -> str:
        self._request_seq += 1
        kind = "async" if async_request else "sync"
        return f"domino-{kind}-e{self._async_epoch}-s{step}-r{self._request_seq}"

    def _record_chunk_request(self, step: int) -> Tuple[Optional[float], Optional[int]]:
        request_time = time.perf_counter()
        interval_s = None
        steps_since = None
        if self._last_chunk_request_time is not None:
            interval_s = request_time - self._last_chunk_request_time
        if self._last_chunk_request_step is not None:
            steps_since = step - self._last_chunk_request_step
        self._last_chunk_request_time = request_time
        self._last_chunk_request_step = step
        return interval_s, steps_since

    def _decode_response_to_actions(
        self,
        response: dict,
        request_state: Optional[np.ndarray],
        request_prev_action: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, dict]:
        response_data = response.get("data", {})
        server_timing = response_data.get("_timing", {}) or {}
        try:
            normalized_actions = response_data["normalized_actions"]  # B, chunk, D
        except KeyError:
            print(f"Response data: {response}")
            raise KeyError(f"Key 'normalized_actions' not found in response data: {response_data.keys()}")

        normalized_actions = normalized_actions[0]
        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions,
            action_norm_stats=self.action_norm_stats,
            normalization_mode=self.normalization_mode,
        )

        if self.action_mode == "delta":
            raw_actions = self._delta_to_absolute(raw_actions, request_state, base_action=request_prev_action)
        elif self.action_mode == "rel":
            raw_actions = self._rel_to_absolute(raw_actions)
        return raw_actions, server_timing

    def _activate_action_chunk(
        self,
        raw_actions: np.ndarray,
        source_step: Optional[int],
        request_id: str,
        activation_step: Optional[int] = None,
    ) -> dict:
        if self.chunk_transition == "replace":
            self.raw_actions = raw_actions
            self._raw_action_cursor = 0
            self._active_chunk_source_step = source_step
            self._active_chunk_request_id = request_id
            return self._empty_handoff_info()
        if self.chunk_transition == "rtc":
            rtc_actions, handoff_info = self._build_rtc_handoff_actions(
                raw_actions=raw_actions,
                source_step=source_step,
                activation_step=activation_step,
            )
            self.raw_actions = rtc_actions
            self._raw_action_cursor = 0
            self._active_chunk_source_step = source_step
            self._active_chunk_request_id = request_id
            return handoff_info
        raise RuntimeError(f"Unsupported chunk_transition: {self.chunk_transition}")

    def _submit_async_request(self, vla_input: dict, state: Optional[np.ndarray], step: int) -> Tuple[bool, Optional[float], Optional[int]]:
        if self.async_worker is None:
            return False, None, None
        interval_s, steps_since = self._record_chunk_request(step)
        request = {
            "request_id": self._next_request_id(step, async_request=True),
            "query": None,
            "step": step,
            "epoch": self._async_epoch,
            "submitted_at_s": time.perf_counter(),
            "state": None if state is None else np.array(state).copy(),
            "prev_action": None if self.prev_action is None else np.array(self.prev_action).copy(),
        }
        request["query"] = {"type": "infer", "request_id": request["request_id"], "payload": vla_input}
        submitted = self.async_worker.submit(request)
        if not submitted:
            return False, interval_s, steps_since
        return True, interval_s, steps_since

    def _request_action_chunk_sync(
        self,
        vla_input: dict,
        state: Optional[np.ndarray],
        step: int,
    ) -> Tuple[np.ndarray, dict, float, str]:
        request_id = self._next_request_id(step, async_request=False)
        websocket_start = time.perf_counter()
        response = self.client.predict_action({"type": "infer", "request_id": request_id, "payload": vla_input})
        websocket_s = time.perf_counter() - websocket_start
        raw_actions, server_timing = self._decode_response_to_actions(
            response=response,
            request_state=None if state is None else np.array(state).copy(),
            request_prev_action=None if self.prev_action is None else np.array(self.prev_action).copy(),
        )
        return raw_actions, server_timing, websocket_s, request_id

    def _poll_async_chunk(self, activation_step: Optional[int] = None) -> Tuple[bool, Optional[dict], Optional[str]]:
        if self.async_worker is None:
            return False, None, None
        result = self.async_worker.pop_latest_result()
        if result is None:
            return False, None, None
        if result.get("epoch") != self._async_epoch:
            return False, None, "stale"
        if not result.get("ok"):
            raise RuntimeError(f"Async policy inference failed: {result.get('error')}")

        raw_actions, server_timing = self._decode_response_to_actions(
            response=result["response"],
            request_state=result.get("state"),
            request_prev_action=result.get("prev_action"),
        )
        chunk = {
            "raw_actions": raw_actions,
            "server_timing": server_timing,
            "websocket_roundtrip_s": result.get("websocket_roundtrip_s"),
            "source_step": result.get("step"),
            "request_id": result.get("request_id"),
            "finished_at_s": result.get("finished_at_s"),
        }
        self.async_result_cache.append(chunk)
        chunk["handoff_info"] = self._activate_action_chunk(
            raw_actions,
            result.get("step"),
            result.get("request_id"),
            activation_step=activation_step,
        )
        return True, chunk, None

    def _wait_for_async_chunk(self, activation_step: Optional[int] = None) -> Optional[dict]:
        if self.async_worker is None or not self.async_worker.has_pending():
            return None
        while self.async_worker.has_pending():
            activated, chunk, _ = self._poll_async_chunk(activation_step=activation_step)
            if activated:
                return chunk
            time.sleep(0.001)
        activated, chunk, _ = self._poll_async_chunk(activation_step=activation_step)
        return chunk if activated else None

    def _normalize_state_for_model(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != CONTINUOUS_ACTION_MASK.shape[0]:
            raise ValueError(f"Expected 14-D robot state, got shape {state.shape}.")
        normalized = self.normalize_state(
            state,
            self.state_norm_stats,
            normalization_mode=self.normalization_mode,
        )
        return normalized.astype(np.float32)[None, :]

    # ---- main step -------------------------------------------------------

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        step_start = time.perf_counter()
        prepare_start = step_start
        state = example.get("state", None)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if task_description != self.task_description:
            if task_description:
                print(
                    "[EVAL_RECORD] "
                    + json.dumps(
                        {
                            "step": step,
                            "instruction": str(task_description),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            self.reset(task_description)
            if self.action_mode in ["delta", "rel"] and state is not None:
                self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in images]
        model_video_frame = self._build_model_video_frame(images)
        model_images = self._build_history_video_frames(model_video_frame)
        if model_images is None:
            model_images = [model_video_frame] if self.multi_view_concat else images

        example_copy = example.copy()
        example_copy["image"] = model_images
        if state is not None:
            example_copy["state"] = self._normalize_state_for_model(state)
        else:
            example_copy.pop("state", None)
        # Use the model video frame as the reference for history.
        if not self.history_as_video_frames:
            example_copy.update(self._build_history_payload(model_video_frame))

        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        action_chunk_size = self.action_chunk_size
        prepare_s = time.perf_counter() - prepare_start
        chunk_request = False
        chunk_activated = False
        async_result_ready = False
        async_result_status = "disabled" if not self.async_inference else "none"
        sync_block = False
        websocket_s = 0.0
        server_timing = {}
        chunk_request_interval_s = None
        steps_since_last_chunk_request = None
        handoff_info = self._empty_handoff_info()
        postprocess_start = time.perf_counter()

        if self.async_inference:
            activated, chunk, status = self._poll_async_chunk(activation_step=step)
            async_result_ready = activated
            if status is not None:
                async_result_status = status
            if activated and chunk is not None:
                chunk_activated = True
                async_result_status = "activated"
                websocket_s = chunk.get("websocket_roundtrip_s") or 0.0
                server_timing = chunk.get("server_timing", {}) or {}
                handoff_info = chunk.get("handoff_info", handoff_info) or handoff_info

            if self.raw_actions is None:
                sync_block = True
                chunk_request = True
                chunk_request_interval_s, steps_since_last_chunk_request = self._record_chunk_request(step)
                raw_actions, server_timing, websocket_s, request_id = self._request_action_chunk_sync(vla_input, state, step)
                handoff_info = self._activate_action_chunk(raw_actions, step, request_id, activation_step=step)
                chunk_activated = True
                async_result_status = "sync_bootstrap"
            elif self._raw_action_cursor >= len(self.raw_actions):
                waited_chunk = self._wait_for_async_chunk(activation_step=step)
                if waited_chunk is None:
                    sync_block = True
                    chunk_request = True
                    chunk_request_interval_s, steps_since_last_chunk_request = self._record_chunk_request(step)
                    raw_actions, server_timing, websocket_s, request_id = self._request_action_chunk_sync(vla_input, state, step)
                    handoff_info = self._activate_action_chunk(raw_actions, step, request_id, activation_step=step)
                    chunk_activated = True
                    async_result_status = "sync_cursor_exhausted"
                else:
                    chunk_activated = True
                    async_result_ready = True
                    async_result_status = "activated_after_wait"
                    websocket_s = waited_chunk.get("websocket_roundtrip_s") or 0.0
                    server_timing = waited_chunk.get("server_timing", {}) or {}
                    handoff_info = waited_chunk.get("handoff_info", handoff_info) or handoff_info

            if not sync_block and self.async_worker is not None and not self.async_worker.has_pending():
                submitted, interval_s, steps_since = self._submit_async_request(vla_input, state, step)
                if submitted:
                    chunk_request = True
                    chunk_request_interval_s = interval_s
                    steps_since_last_chunk_request = steps_since
        else:
            chunk_request = step % action_chunk_size == 0 or self.raw_actions is None
            if chunk_request:
                chunk_request_interval_s, steps_since_last_chunk_request = self._record_chunk_request(step)
                raw_actions, server_timing, websocket_s, request_id = self._request_action_chunk_sync(vla_input, state, step)
                handoff_info = self._activate_action_chunk(raw_actions, step, request_id, activation_step=step)
                chunk_activated = True

        # --- Push current frame into history buffer (after inference) ---
        self._push_history_frame(model_video_frame)

        action_idx = self._raw_action_cursor if self.async_inference else step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            action_idx = len(self.raw_actions) - 1

        current_action = self.raw_actions[action_idx]
        if self.async_inference:
            self._raw_action_cursor += 1

        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        current_action = current_action[MODEL_TO_ENV_ACTION_ORDER]
        postprocess_s = time.perf_counter() - postprocess_start
        self.last_timing = {
            "step_total_s": time.perf_counter() - step_start,
            "client_prepare_s": prepare_s,
            "websocket_roundtrip_s": websocket_s,
            "client_postprocess_s": postprocess_s,
            "server_total_s": server_timing.get("server_total_s"),
            "server_model_inference_s": server_timing.get("model_inference_s"),
            "server_action_head_s": server_timing.get("action_head_s"),
            "server_overhead_s": server_timing.get("server_overhead_s"),
            "chunk_request": float(chunk_request),
            "chunk_request_interval_s": chunk_request_interval_s,
            "steps_since_last_chunk_request": steps_since_last_chunk_request,
            "action_chunk_size": float(action_chunk_size),
            "async_inference": float(self.async_inference),
            "async_pending": float(self.async_worker.has_pending() if self.async_worker is not None else False),
            "async_result_ready": float(async_result_ready),
            "async_result_status": async_result_status,
            "sync_block": float(sync_block),
            "chunk_activated": float(chunk_activated),
            "chunk_transition": self.chunk_transition,
            "active_chunk_source_step": self._active_chunk_source_step,
            "active_chunk_cursor": float(action_idx),
            "rtc_applied": float(bool(handoff_info.get("rtc_applied", False))),
            "rtc_overlap_steps": handoff_info.get("rtc_overlap_steps"),
            "rtc_inference_delay_steps": handoff_info.get("rtc_inference_delay_steps"),
            "rtc_new_start_index": handoff_info.get("rtc_new_start_index"),
            "rtc_blend_dim_count": handoff_info.get("rtc_blend_dim_count"),
            "rtc_old_weight_min": handoff_info.get("rtc_old_weight_min"),
            "rtc_old_weight_max": handoff_info.get("rtc_old_weight_max"),
        }
        return current_action

    # ---- normalization helpers -------------------------------------------

    @staticmethod
    def normalize_state(
        state: np.ndarray,
        state_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> np.ndarray:
        continuous_mask = CONTINUOUS_ACTION_MASK
        state_high, state_low = ModelClient._get_normalization_bounds(
            state_norm_stats, normalization_mode=normalization_mode
        )
        valid_mask = continuous_mask & (state_high != state_low)
        normalized_state = np.where(
            valid_mask,
            (state - state_low) / (state_high - state_low) * 2 - 1,
            state,
        )
        normalized_state = np.where(
            ~continuous_mask,
            (normalized_state > 0.5).astype(normalized_state.dtype),
            normalized_state,
        )
        return normalized_state

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> np.ndarray:
        action_high, action_low = ModelClient._get_normalization_bounds(
            action_norm_stats, normalization_mode=normalization_mode
        )
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
        normalized_actions = np.clip(normalized_actions, -1, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        actions = np.where(
            mask,
            actions,
            (normalized_actions > 0.5).astype(actions.dtype),
        )
        return actions

    def _delta_to_absolute(
        self,
        delta_actions: np.ndarray,
        current_state: np.ndarray,
        base_action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        abs_actions = np.zeros_like(delta_actions)
        mask = self.action_norm_stats.get("mask", np.ones(delta_actions.shape[-1], dtype=bool))
        base = current_state if current_state is not None else base_action
        if base is None:
            base = self.prev_action if self.prev_action is not None else self.initial_state
        if base is None:
            raise ValueError("delta action conversion requires current_state, initial_state, or prev_action")
        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        abs_actions = np.zeros_like(rel_actions)
        mask = self.action_norm_stats.get("mask", np.ones(rel_actions.shape[-1], dtype=bool))
        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(mask, rel_actions[i] + self.initial_state, rel_actions[i])
        return abs_actions

    # ---- stats / config helpers ------------------------------------------

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path, action_mode: str = "abs") -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        stats = norm_stats[unnorm_key]
        stats_action_mode = stats.get("action_mode")
        if action_mode == "abs" and stats_action_mode not in (None, "abs"):
            raise ValueError(
                f"Statistics key `{unnorm_key}` was saved for action_mode=`{stats_action_mode}`, "
                "but eval requested action_mode=`abs`."
            )

        if action_mode in stats:
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        if "action" in stats:
            if action_mode != "abs":
                raise ValueError(
                    f"Statistics key `{unnorm_key}` only provides `abs` action stats, "
                    f"but action_mode=`{action_mode}` was requested."
                )
            return stats["action"]
        raise ValueError(
            f"Invalid statistics file format for key `{unnorm_key}`. "
            f"Available top-level keys: {sorted(stats.keys())}"
        )

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        available_keys = sorted(norm_stats.keys())
        if unnorm_key is None:
            if len(available_keys) == 1:
                return available_keys[0]
            raise ValueError(
                "`unnorm_key` must be provided when multiple normalization statistics are available. "
                f"Available keys: {available_keys}"
            )
        if unnorm_key not in norm_stats:
            raise KeyError(f"Unknown `unnorm_key`: `{unnorm_key}`. Available keys: {available_keys}")
        return unnorm_key

    @staticmethod
    def _get_normalization_bounds(
        norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> tuple[np.ndarray, np.ndarray]:
        if normalization_mode == "q99":
            if "q01" not in norm_stats or "q99" not in norm_stats:
                raise KeyError("Normalization mode `q99` requires statistics keys `q01` and `q99`.")
            return np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        if normalization_mode == "min_max":
            if "min" not in norm_stats or "max" not in norm_stats:
                raise KeyError("Normalization mode `min_max` requires statistics keys `min` and `max`.")
            return np.array(norm_stats["max"]), np.array(norm_stats["min"])
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}. Expected one of ['min_max', 'q99'].")


# ---------------------------------------------------------------------------
# DOMINO eval_policy.py API
# ---------------------------------------------------------------------------

def get_model(usr_args: dict) -> ModelClient:
    """Construct a ``ModelClient`` from the merged DOMINO config dict.

    All keys in ``deploy_policy.yml`` plus ``--overrides`` appear in
    *usr_args*.  History settings are optional and default to disabled.
    """
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 5694),
        unnorm_key=usr_args.get("unnorm_key", None),
        action_mode=usr_args.get("action_mode", "abs"),
        normalization_mode=usr_args.get(
            "action_normalization_mode",
            usr_args.get("normalization_mode", "min_max"),
        ),
        image_size=usr_args.get("image_size", None),
        # History (default: disabled)
        history_k=int(usr_args.get("history_k", 0)),
        history_stride=int(usr_args.get("history_stride", 1)),
        history_mode=usr_args.get("history_mode", "flow"),
        history_as_video_frames=_as_bool(usr_args.get("history_as_video_frames", False)),
        multi_view_concat=_as_bool(usr_args.get("multi_view_concat", False)),
        history_image_size=usr_args.get("history_image_size", None),
        history_flow_compute_size=usr_args.get("history_flow_compute_size", None),
        # Async inference / chunk handoff
        async_inference=_as_bool(usr_args.get("async_inference", True)),
        chunk_transition=usr_args.get("chunk_transition", "rtc"),
        async_result_cache_size=int(usr_args.get("async_result_cache_size", 2)),
        rtc_execution_horizon=int(usr_args.get("rtc_execution_horizon", 10)),
        rtc_max_guidance_weight=float(usr_args.get("rtc_max_guidance_weight", 10.0)),
        rtc_prefix_attention_schedule=usr_args.get("rtc_prefix_attention_schedule", "exp"),
        rtc_inference_delay=usr_args.get("rtc_inference_delay", None),
        rtc_compensate_inference_delay=_as_bool(usr_args.get("rtc_compensate_inference_delay", True)),
        rtc_blend_dims=usr_args.get("rtc_blend_dims", None),
    )


def reset_model(model: ModelClient) -> None:
    model.reset(task_description="")


def eval(TASK_ENV, model: ModelClient, observation: dict) -> None:
    eval_start = time.perf_counter()
    observation_start = eval_start
    instruction = TASK_ENV.get_instruction()

    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    images = [head_img, left_img, right_img]  # [head, left_wrist, right_wrist]
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    if state.shape[0] != ENV_TO_MODEL_ACTION_ORDER.shape[0]:
        raise ValueError(f"Expected 14-D DOMINO joint_action vector, got shape {state.shape}.")
    state = state[ENV_TO_MODEL_ACTION_ORDER]

    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,
    }
    observation_s = time.perf_counter() - observation_start

    step_idx = TASK_ENV.take_action_cnt
    model_start = time.perf_counter()
    action = model.step(example, step=step_idx)
    model_step_s = time.perf_counter() - model_start

    take_action_start = time.perf_counter()
    TASK_ENV.take_action(action)
    take_action_s = time.perf_counter() - take_action_start
    eval_total_s = time.perf_counter() - eval_start

    timing = getattr(model, "last_timing", {}) or {}

    def fmt_ms(value: Optional[float]) -> str:
        return "NA" if value is None else f"{value * 1000.0:.3f}"

    def fmt_value(value: object) -> str:
        return "NA" if value is None else str(value)

    def fmt_float(value: Optional[float]) -> str:
        return "NA" if value is None else f"{float(value):.3f}"

    print(
        "[TIMING_EXEC] "
        f"step={step_idx} "
        f"chunk_request={int(bool(timing.get('chunk_request', 0.0)))} "
        f"chunk_request_interval_ms={fmt_ms(timing.get('chunk_request_interval_s'))} "
        f"steps_since_last_chunk_request={timing.get('steps_since_last_chunk_request', 'NA')} "
        f"eval_total_ms={fmt_ms(eval_total_s)} "
        f"observation_to_example_ms={fmt_ms(observation_s)} "
        f"model_step_ms={fmt_ms(model_step_s)} "
        f"client_prepare_ms={fmt_ms(timing.get('client_prepare_s'))} "
        f"websocket_roundtrip_ms={fmt_ms(timing.get('websocket_roundtrip_s'))} "
        f"server_total_ms={fmt_ms(timing.get('server_total_s'))} "
        f"server_model_inference_ms={fmt_ms(timing.get('server_model_inference_s'))} "
        f"server_action_head_ms={fmt_ms(timing.get('server_action_head_s'))} "
        f"server_overhead_ms={fmt_ms(timing.get('server_overhead_s'))} "
        f"client_postprocess_ms={fmt_ms(timing.get('client_postprocess_s'))} "
        f"take_action_ms={fmt_ms(take_action_s)} "
        f"action_chunk_size={int(timing.get('action_chunk_size', model.action_chunk_size))} "
        f"async_inference={int(bool(timing.get('async_inference', 0.0)))} "
        f"async_pending={int(bool(timing.get('async_pending', 0.0)))} "
        f"async_result_ready={int(bool(timing.get('async_result_ready', 0.0)))} "
        f"sync_block={int(bool(timing.get('sync_block', 0.0)))} "
        f"chunk_activated={int(bool(timing.get('chunk_activated', 0.0)))} "
        f"chunk_transition={timing.get('chunk_transition', 'NA')} "
        f"active_chunk_source_step={timing.get('active_chunk_source_step', 'NA')} "
        f"active_chunk_cursor={int(timing.get('active_chunk_cursor', -1))} "
        f"async_result_status={timing.get('async_result_status', 'NA')} "
        f"rtc_applied={int(bool(timing.get('rtc_applied', 0.0)))} "
        f"rtc_overlap_steps={fmt_value(timing.get('rtc_overlap_steps'))} "
        f"rtc_inference_delay_steps={fmt_value(timing.get('rtc_inference_delay_steps'))} "
        f"rtc_new_start_index={fmt_value(timing.get('rtc_new_start_index'))} "
        f"rtc_blend_dim_count={fmt_value(timing.get('rtc_blend_dim_count'))} "
        f"rtc_old_weight_min={fmt_float(timing.get('rtc_old_weight_min'))} "
        f"rtc_old_weight_max={fmt_float(timing.get('rtc_old_weight_max'))}",
        flush=True,
    )
