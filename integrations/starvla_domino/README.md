# StarVLA DOMINO Async Integration

This directory preserves the deployment-side implementation used by the saved
DOMINO evaluation. It was copied from the local StarVLA repository at commit
`db8fe59` plus the working-tree configuration present on 2026-07-15.

The relevant mechanisms are:

- `_AsyncInferenceWorker`: one in-flight websocket request outside the control loop;
- activation of the newest completed action chunk;
- RTC handoff with delay compensation and weighted old/new chunk overlap;
- per-step timing for requests, model inference, action heads, staleness, and blocking.

The saved `18/100` run enabled `async_inference=true` through its runtime
evaluation override. The copied YAML reflects a later diagnostic state with the
flag disabled; set it to `true` to reproduce the asynchronous mode.

This is an integration snapshot, not a standalone executable. It depends on
StarVLA, DOMINO, and their runtime environment. Upstream StarVLA code is MIT
licensed; its license is retained as `LICENSE.starvla`.

Lightweight evaluation summaries are in `results/async_domino_eval/`. They
compare two system configurations, not a matched one-variable ablation.
