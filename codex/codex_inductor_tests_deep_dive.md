# TorchInductor test suite × `torch.compile`: stage-by-stage mapping

> Scope: map **PyTorch’s `test/inductor`** tests to the **exact TorchDynamo → AOTAutograd → Inductor** pipeline stages they exercise, with links and tiny code excerpts (<25 words each). Organized in the same order that Inductor lowers models. This doc complements `codex/vllm-compilation` with upstream test coverage and actionable entrypoints for LLM-focused tuning (extra decompositions, passes, novel ops, or cudagraph plumbing).

_Updated: 2025‑09‑08_

---

## 0) Pipeline refresher (what we’re testing)
1. **Dynamo capture & guards** → bytecode trampoline, FX graph, guard set, dynamic shapes.
2. **Functionalization & decompositions** → PrimTorch, mutate→functional.
3. **FX post-grad graph shaping** → partitioning, pattern rewrites.
4. **Inductor IR & fusion** → scheduler, loop IR, Triton/C++ lowering.
5. **Codegen & autotune** → kernel emission, CUTLASS/Triton search.
6. **Runtime** → cudagraph trees, streams, collectives.
7. **Caching & hot-load** → Dynamo/Inductor caches & rehydration.

---

## 1) Core harness used across many tests: `test_torchinductor.py`
**File**: `test/inductor/test_torchinductor.py` (huge umbrella test module)

- **Global config hooks** (turns on debug, index asserts, etc.)
  > `config.patch({"debug": True, ... "generate_intermediate_hooks": True})` (class setup). citeturn8search5

- **Capturing the _post‑grad_ FX graph**
  > `logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")` then `f(*inputs)` to collect. citeturn8search5

- **Unified runner**
  > `check_model(...): eager vs compiled; reset Dynamo; compare dtypes/strides/gradients; can fetch Triton code.` citeturn8search5

- **Compilation entry**
  > Imports `compile_fx`, `compile_fx_inner`, and helpers like `run_and_get_triton_code`. citeturn8search5

**What stages it isolates**
- **Dynamo**: resets state per test; exercises `torch.compile` with device/dtype matrix.
- **FX post‑grad**: uses log channel to assert expected patterns after functionalization.
- **Fusion/codegen**: verifies emitted kernels & correctness across ops/dtypes.

> Use this file as a **reference harness**: it shows how to force logs for each phase, and where to poke Inductor’s IR with minimal boilerplate. citeturn8search5

---

## 2) Stage ⇒ Tests

### 2.1 Dynamo capture & guards
**Goals:** minimize recompiles, validate guard precision, exercise dynamic shapes, and ensure no spurious graph breaks.

- **`test_torchinductor.py` — reset & compile discipline**
  > `torch._dynamo.reset(); torch._inductor.metrics.reset()` in `setUp/tearDown`. Ensures per‑test clean capture/metrics. citeturn8search5
- **`test_codecache.py` — cache artifacts and hot load**
  - Counts frames with a **backend counter**:
    > `CompileCounterWithBackend("inductor")` + `@torch.compile(..., fullgraph=True)` fixture. citeturn6search0
  - Records/loads artifacts:
    > `torch.compiler.save_cache_artifacts()` / `load_cache_artifacts(artifact_bytes)`. citeturn6search0
  - Asserts FX‑graph cache counters:
    > `counters["inductor"]["fxgraph_cache_{miss,hit}"]` increments as expected. citeturn6search0
- **Control flow / guards** (referenced in CI/issue):
  > `test_control_flow.py::test_cond_control_flow_with_precomputed_size` was added to stress guarded branches. citeturn5search3

**Why this matters for LLMs**
- Use **stance** + counters to catch recompiles before they become tail‑latency spikes. Bucket sequence lengths to stabilize guards.

---

### 2.2 Functionalization & decompositions
**Goals:** guarantee prim decomposition coverage and mutation‑free graphs so Inductor can fuse.

- **In‐file (umbrella)**: post‑grad logs
  > `get_post_grad_graph(f, inputs)` captures the FX graph **after** AOT/functionalization. Perfect to assert that, say, `view+copy` was normalized before fusion. citeturn8search5
- **Decomposition correctness (umbrella)**
  > Imports `lowering`, `pad_mm`, etc., which trigger specific decompositions during compile. citeturn8search5

**Why this matters for LLMs**
- Custom **KV‑layout** or **attention bias** decompositions can move complexity out of runtime, enabling larger fusions.

---

### 2.3 FX post‑grad shaping & partitioning
**Goals:** ensure the FX graph is in the shape Inductor expects; validate pattern rewrites.

- **Logging post‑grad graphs**
  > The `logs_to_string(..., "post_grad_graphs")` channel is the exact hook Inductor exposes for verifying this stage. citeturn8search5

- **Device & dtype matrices**
  > Parametrized dtypes and devices enforce the FX graph is portable across backends (e.g., BF16 path on SM80+). citeturn8search5

**LLM optimization ideas**
- Introduce a **rewrite pass** that canonicalizes attention blocks (e.g., fold `silu*mul` prior to GEMM) and assert it appears in post‑grad logs.

---

### 2.4 Inductor IR & fusion
**Goals:** scheduler legality, loop IR, fusion groups, and emitted kernel parity.

- **Umbrella file codegen queries**
  > Helpers like `run_and_get_triton_code`, `run_and_get_kernels` inspect emitted kernels for specific ops. citeturn8search5

- **Max‑autotune suite (referenced)**
  > `test_max_autotune.py` is dedicated to autotune/fusion; recent CUDA 12.6/H100 reports catalog failures and shape‑specialization quirks. Use them as regression probes. citeturn5search0turn5search4

- **Operator parity**
  > `test_torchinductor_opinfo.py` failures show missing decompositions/fusions on certain dtypes/devices; they are the first bellwether. citeturn0search1

**LLM optimization ideas**
- Pre‑shape Q/K/V to avoid **transpose‑heavy** paths that defeat fusion; confirm via Triton source diffs collected by the helpers above.

---

### 2.5 Codegen & autotune (Triton/CUTLASS)
**Goals:** ensure autotune explores the right search space and that backend selection is sane.

- **CUTLASS / autotune CI threads**
  > H100/ROCm accuracy and autotune regressions appear in `test_max_autotune.py` and related issues; track when changing tile heuristics. citeturn4search3turn0search19

- **Docs**
  > Inductor profiling docs show how to diff kernel perf and run kernels standalone. Useful when tuning LLM matmuls. citeturn0search20

**LLM optimization ideas**
- Pre‑warm autotune **per decode bucket**; bias tile search toward KV‑friendly strides.

---

### 2.6 Runtime: CUDA graphs & collectives
**Goals:** validate cudagraph trees, NCCL boundaries, and side‑effects across calls.

- **CUDAGraph behavior issues**
  > Default‑on cudagraph RFC and edge cases (detached outputs, batch changes). These influence tree policies your runtime must respect. citeturn0search2turn0search11

- **Output lifetime gotchas**
  > Reports that subsequent runs can overwrite outputs if buffers aren’t managed correctly; track when designing replay pools. citeturn0search16

**LLM optimization ideas**
- Keep collectives **outside capture** or use tree‑aware entrypoints; pre‑allocate persistent I/O to stabilize addresses across replays.

---

### 2.7 Caching & hot‑load
**Goals:** exercise Inductor & PGO caches, FX‑graph cache, artifact bundling and reload.

- **`test_codecache.py` (rich examples)**
  - Fullgraph cache miss/hit counters
    > `fxgraph_cache_{miss,hit}` count changes across shapes. citeturn6search0
  - Artifact round‑trip
    > `save_cache_artifacts()` → bytes blob → `load_cache_artifacts(...)` hot‑loads functions. citeturn6search0
  - PGO swap & generic caches
    > Registers arbitrary cache artifacts via `CacheArtifactFactory.register`. Shows the generic hot‑load API surface. citeturn6search0
  - Guard behavior in cache lookups
    > `test_cache_guard` demonstrates guard divergence leading to a fresh compile. citeturn6search0

**LLM optimization ideas**
- Capture and ship **pre‑tuned**, **bucket‑specialized** artifacts (kernels + PGO profiles) for your top SKUs; verify parity with hot‑load tests.

---

## 3) Other notable test families (brief)
- **Compiled autograd**
  > `test_compiled_autograd.py` focuses on compiled backward partitions and fake‑tensor mode correctness; CI often flags verbosity/graph mismatch failures. citeturn7search4turn8search6
- **AOTInductor**
  > `test_aot_inductor.py` checks AOT ABI and model export; known gaps with ops like `torch.select` documented with a proposed test. citeturn7search6turn5search1
- **Control flow**
  > `test_control_flow.py` targets Dynamo’s graph‑break/capture coverage for branches/loops; occasionally trips platform decorators. citeturn5search3

---

## 4) API surface & entrypoints (documented and semi‑undocumented)
- **Dynamo/compile**: `torch.compile`, `torch.compiler.set_stance`, `torch._dynamo.reset`, `TORCH_LOGS=recompiles` for diagnostics. Docs: dynamic shapes. citeturn0search4turn0search23
- **Inductor config**: `torch._inductor.config.patch({...})` toggles IR/debug/autotune.
- **FX post‑grad logs**: `logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")` is the sanctioned hook to inspect FX after functionalization. citeturn8search5
- **Codegen inspection**: `run_and_get_triton_code`, `run_and_get_kernels`, `run_and_get_cpp_code`. citeturn8search5
- **Caching**: `torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()` + counters in `torch._inductor.counters`. citeturn6search0

---

## 5) vLLM vs upstream tests (short contrast)
- vLLM adds **product‑shaped** tests (piecewise capture, attention/quant fusions, TP/SP) and a **Dynamo artifact hot‑start** test; upstream focuses on **platform‑shaped** coverage (opinfo, autotune backends, compiled autograd, AOT ABI).

---

## 6) Practical recipes (how to replicate these checks yourself)
- **Assert no recompiles**: wrap micro‑model in `CompileCounterWithBackend` and set stance; verify `frame_count`/counters.
- **Inspect post‑grad FX**: replicate `get_post_grad_graph`; assert your pass injected/folded the intended pattern.
- **Check fusion/codegen**: record Triton source and grep for expected kernels or tile configs.
- **Cache round‑trip**: use `save_cache_artifacts`/`load_cache_artifacts` to pre‑seed warm starts.

---

### Appendix A — small code anchors
- Post‑grad log capture:
  > `log_stream, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")` citeturn8search5
- Test harness config patch:
  > `config.patch({"debug": True, "debug_index_asserts": True, ...})` citeturn8search5
- Cache artifacts round‑trip:
  > `artifacts = torch.compiler.save_cache_artifacts(); load_cache_artifacts(artifact_bytes)` citeturn6search0
- Compile counter backend:
  > `backend = CompileCounterWithBackend("inductor"); @torch.compile(backend=backend)` citeturn6search0

---

> Questions to investigate next: map **specific tests** (e.g., `test_max_autotune_decompose_k_dynamic_input`) to the Triton configs they generate and identify which KV‑friendly tile heuristics reduce decode P50/P99.

