TORCH_TRACE in torch.compile — What It Records and Where It’s Wired Up

- TORCH_TRACE enablement
  - `torch/_logging/_internal.py:51` defines the env var name: `TRACE_ENV_VAR = "TORCH_TRACE"`.
  - `torch/_logging/_internal.py:1081` reads `TORCH_TRACE` in `_init_logs()` and attaches a lazy trace handler to a dedicated logger `torch.__trace`.
  - `torch/_logging/_internal.py:1106` defines `LazyTraceHandler`, which only opens a file on the first trace event; files are created under the directory from `TORCH_TRACE` with prefix `dedicated_log_torch_trace_...` and suffix `.log` (`torch/_logging/_internal.py:1175`).
  - `torch/_logging/_internal.py:1305` defines `trace_structured(...)`, which is the API all components use to emit structured events. It short‑circuits unless the trace handler has been installed: see the `if trace_log.handlers:` guard at `torch/_logging/_internal.py:1343`.
  - `torch/_logging/structured.py:1` contains helpers used by structured tracing. In particular:
    - `intern_string()` emits `str` intern events (`torch/_logging/structured.py:18`).
    - `dump_file()` emits file payloads for evaluated code (`torch/_logging/structured.py:29`).

- Relationship to TORCH_LOGS
  - `docs/source/compile/programming_model.observability.md:29` notes tlparse/TORCH_TRACE gather their content from torch._logging; structured trace entries are produced via `trace_structured(...)` throughout the stack and written to the dedicated trace files when `TORCH_TRACE` is set.

- Dynamo (front‑end) artifacts and events
  - Start of a frame compile: emits `dynamo_start` with a compact stack summary.
    - `torch/_dynamo/convert_frame.py:249` (`torch._logging.trace_structured("dynamo_start", ...)`).
  - Output FX graph per frame: emits `dynamo_output_graph` with tensor size metadata and the readable FX graph as payload.
    - `torch/_dynamo/output_graph.py:1923`.
  - Guard summary (C++/manager rendering) and timing: string payload of guard structure and measured latency.
    - `torch/_dynamo/guards.py:3458` (`trace_structured("dynamo_cpp_guards_str", ...)`).
  - Graph break reason captured as an artifact named `dynamo_graph_break_reason`.
    - `torch/_dynamo/exc.py:441` and `torch/_dynamo/exc.py:481` (payload is the verbose message/stack).
  - Precompile packaging bypass (when applicable): artifact `precompile_cache_bypass` with JSON payload.
    - `torch/_dynamo/output_graph.py:1723`.

- AOTAutograd artifacts and events
  - Forward/backward graph snapshots and metadata (when AOT logging is enabled):
    - `aot_forward_graph_fw_metadata` (string dataclass repr): `torch/_functorch/_aot_autograd/graph_capture.py:301` and cache path `torch/_functorch/_aot_autograd/autograd_cache.py:837`.
    - `aot_forward_graph_fw_subclass_metadata` (string): `torch/_functorch/_aot_autograd/graph_capture.py:312` and cache path `autograd_cache.py:845`.
    - `aot_inference_graph` or `aot_forward_graph` (FX readable string): `torch/_functorch/_aot_autograd/graph_capture.py:320` and cache path `autograd_cache.py:860`.
    - `aot_backward_graph`: `torch/_functorch/_aot_autograd/autograd_cache.py:866`.
  - Saved‑tensors hooks instrumentation (when enabled):
    - `aot_forward_graph_pre_saved_tensors_hooks`: `torch/_functorch/_aot_autograd/graph_compile.py:962`.
    - `aot_backward_graph_pre_saved_tensors_hooks`: `graph_compile.py:969`.
    - `aot_forward_graph_pre_assert_saved_tensors_hooks`: `graph_compile.py:1296`.
    - `aot_backward_graph_pre_assert_saved_tensors_hooks`: `graph_compile.py:1303`.
    - Aggregated textual log `aot_saved_tensors_hooks_graphs`: `graph_compile.py:1275`.

- Inductor artifacts and events
  - Runnable graph string (Python runnable repro): artifact `fx_graph_runnable`.
    - In fresh compiles: `torch/_inductor/compile_fx.py:1215`.
    - On cache hits: `torch/_inductor/codecache.py:1220`.
  - FX graph snapshots around passes (for provenance/highlighting):
    - `before_pre_grad_graph`: `torch/_inductor/compile_fx.py:2322`.
    - `after_pre_grad_graph`: `torch/_inductor/compile_fx.py:2354`.
    - `before_post_grad_graph`: `torch/_inductor/compile_fx.py:1266`.
    - `after_post_grad_graph`: `torch/_inductor/compile_fx.py:1305`.
    - `before_joint_graph` (inference partitioning): `torch/_inductor/compile_fx.py:2186`.
    - `after_joint_graph`: `torch/_inductor/compile_fx.py:2201`.
    - `inductor_post_grad_graph` on cache hit: `torch/_inductor/codecache.py:1228`.
  - Generated output code source and location:
    - `inductor_output_code` (payload is code; metadata includes `filename`): `torch/_inductor/codecache.py:1232`.
  - Cache action artifacts:
    - `fx_graph_cache_{hit|miss|bypass|disabled}` artifact with JSON details: `torch/_inductor/compile_fx.py:1056`.
  - Provenance and kernel stack traces (when enabled):
    - `inductor_provenance_tracking_node_mappings`: `torch/_inductor/codecache.py:1237`.
    - `inductor_provenance_tracking_kernel_stack_traces`: `torch/_inductor/codecache.py:1245`.
  - Scheduler/overlap debugging:
    - `scheduler_nodes_before_comm_overlap`: `torch/_inductor/scheduler.py:2187`.
    - Collective schedule: `inductor_collective_schedule` at `torch/_inductor/debug.py:725`.
  - Runtime/tensor meta and execution order (for tlparse performance panes):
    - `inductor_runtime_and_tensor_meta` (per‑op estimated runtimes and output meta): `torch/_inductor/debug.py:804`.
    - `graph_execution` (execution order): `torch/_inductor/debug.py:821`.

- Other structured events seen during compilation
  - `inductor_input_graph` (commented in some flows): `torch/_inductor/compile_fx.py:1227`.
  - String interning and file dumps used by the stack trie and source views:
    - `str`: `torch/_logging/structured.py:18`.
    - `dump_file`: `torch/_logging/structured.py:29`.

- How TORCH_TRACE triggers the writes
  - Setting `TORCH_TRACE=/path/to/dir` configures `_init_logs()` to install `LazyTraceHandler` (`torch/_logging/_internal.py:1081`).
  - All `torch._logging.trace_structured(...)` calls across Dynamo, AOTAutograd, and Inductor become active because `trace_log.handlers` is non‑empty (`torch/_logging/_internal.py:1343`).
  - Each call writes a JSON metadata record and a string/JSON payload line to a dedicated log file under the directory you specified (`torch/_logging/_internal.py:1170`). These files are what `tlparse` reads.

Notes
- Artifact names listed above are the `metadata["name"]` attached to the `artifact` structured event, with payloads typically being readable FX graphs, generated code, or JSON blobs. Non‑artifact events like `dynamo_start`, `dynamo_output_graph`, `inductor_output_code`, etc., are separate event types parsed by tlparse.
- Many traditional `TORCH_LOGS` artifact loggers (e.g., `graph_breaks`, `guards`, `output_code`) also exist, but tlparse content comes from the structured trace events, not from stderr. `docs/source/compile/programming_model.observability.md:62` clarifies that `TORCH_LOGS` controls which components emit messages; the structured tracing mirrors key artifacts regardless of console logging.

