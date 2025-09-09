# TorchInductor test suite × `torch.compile`: stage-by-stage mapping

> Scope: map **PyTorch’s `test/inductor`** tests to the **exact TorchDynamo → AOTAutograd → Inductor** pipeline stages they exercise, with links and tiny code excerpts (<25 words each). Organized in the same order that Inductor lowers models. This doc complements `codex/vllm-compilation` with upstream test coverage and actionable entrypoints for LLM-focused tuning (extra decompositions, passes, novel ops, or cudagraph plumbing).

*Updated: 2025‑09‑08*

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

* **Global config hooks** (turns on debug, index asserts, etc.)

  > `config.patch({"debug": True, ... "generate_intermediate_hooks": True})` (class setup).

* **Capturing the *post‑grad* FX graph**

  > `logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")` then `f(*inputs)` to collect.

* **Unified runner**

  > `check_model(...): eager vs compiled; reset Dynamo; compare dtypes/strides/gradients; can fetch Triton code.`

* **Compilation entry**

  > Imports `compile_fx`, `compile_fx_inner`, and helpers like `run_and_get_triton_code`.

**What stages it isolates**

* **Dynamo**: resets state per test; exercises `torch.compile` with device/dtype matrix.
* **FX post‑grad**: uses log channel to assert expected patterns after functionalization.
* **Fusion/codegen**: verifies emitted kernels & correctness across ops/dtypes.

> Use this file as a **reference harness**: it shows how to force logs for each phase, and where to poke Inductor’s IR with minimal boilerplate.

---

## 2) Stage ⇒ Tests

### 2.1 Dynamo capture & guards

**Goals:** minimize recompiles, validate guard precision, exercise dynamic shapes, and ensure no spurious graph breaks.

* **`test_torchinductor.py` — reset & compile discipline**

  > `torch._dynamo.reset(); torch._inductor.metrics.reset()` in `setUp/tearDown`. Ensures per‑test clean capture/metrics.
* **`test_codecache.py` — cache artifacts and hot load**

  * Counts frames with a **backend counter**:

    > `CompileCounterWithBackend("inductor")` + `@torch.compile(..., fullgraph=True)` fixture.
  * Records/loads artifacts:

    > `torch.compiler.save_cache_artifacts()` / `load_cache_artifacts(artifact_bytes)`.
  * Asserts FX‑graph cache counters:

    > `counters["inductor"]["fxgraph_cache_{miss,hit}"]` increments as expected.
* **Control flow / guards** (referenced in CI/issue):

  > `test_control_flow.py::test_cond_control_flow_with_precomputed_size` was added to stress guarded branches.

**Why this matters for LLMs**

* Use **stance** + counters to catch recompiles before they become tail‑latency spikes. Bucket sequence lengths to stabilize guards.

---

### 2.2 Functionalization & decompositions

**Goals:** guarantee prim decomposition coverage and mutation‑free graphs so Inductor can fuse.

* **In‐file (umbrella)**: post‑grad logs

  > `get_post_grad_graph(f, inputs)` captures the FX graph **after** AOT/functionalization. Perfect to assert that, say, `view+copy` was normalized before fusion.
* **Decomposition correctness (umbrella)**

  > Imports `lowering`, `pad_mm`, etc., which trigger specific decompositions during compile.

**Why this matters for LLMs**

* Custom **KV‑layout** or **attention bias** decompositions can move complexity out of runtime, enabling larger fusions.

---

### 2.3 FX post‑grad shaping & partitioning

**Goals:** ensure the FX graph is in the shape Inductor expects; validate pattern rewrites.

* **Logging post‑grad graphs**

  > The `logs_to_string(..., "post_grad_graphs")` channel is the exact hook Inductor exposes for verifying this stage.

* **Device & dtype matrices**

  > Parametrized dtypes and devices enforce the FX graph is portable across backends (e.g., BF16 path on SM80+).

**LLM optimization ideas**

* Introduce a **rewrite pass** that canonicalizes attention blocks (e.g., fold `silu*mul` prior to GEMM) and assert it appears in post‑grad logs.

---

### 2.4 Inductor IR & fusion

**Goals:** scheduler legality, loop IR, fusion groups, and emitted kernel parity.

* **Umbrella file codegen queries**

  > Helpers like `run_and_get_triton_code`, `run_and_get_kernels` inspect emitted kernels for specific ops.

* **Max‑autotune suite (referenced)**

  > `test_max_autotune.py` is dedicated to autotune/fusion; recent CUDA 12.6/H100 reports catalog failures and shape‑specialization quirks. Use them as regression probes.

* **Operator parity**

  > `test_torchinductor_opinfo.py` failures show missing decompositions/fusions on certain dtypes/devices; they are the first bellwether.

**LLM optimization ideas**

* Pre‑shape Q/K/V to avoid **transpose‑heavy** paths that defeat fusion; confirm via Triton source diffs collected by the helpers above.

---

### 2.5 Codegen & autotune (Triton/CUTLASS)

**Goals:** ensure autotune explores the right search space and that backend selection is sane.

* **CUTLASS / autotune CI threads**

  > H100/ROCm accuracy and autotune regressions appear in `test_max_autotune.py` and related issues; track when changing tile heuristics.

* **Docs**

  > Inductor profiling docs show how to diff kernel perf and run kernels standalone. Useful when tuning LLM matmuls.

**LLM optimization ideas**

* Pre‑warm autotune **per decode bucket**; bias tile search toward KV‑friendly strides.

---

### 2.6 Runtime: CUDA graphs & collectives

**Goals:** validate cudagraph trees, NCCL boundaries, and side‑effects across calls.

* **CUDAGraph behavior issues**

  > Default‑on cudagraph RFC and edge cases (detached outputs, batch changes). These influence tree policies your runtime must respect.

* **Output lifetime gotchas**

  > Reports that subsequent runs can overwrite outputs if buffers aren’t managed correctly; track when designing replay pools.

**LLM optimization ideas**

* Keep collectives **outside capture** or use tree‑aware entrypoints; pre‑allocate persistent I/O to stabilize addresses across replays.

---

### 2.7 Caching & hot‑load

**Goals:** exercise Inductor & PGO caches, FX‑graph cache, artifact bundling and reload.

* **`test_codecache.py` (rich examples)**

  * Fullgraph cache miss/hit counters

    > `fxgraph_cache_{miss,hit}` count changes across shapes.
  * Artifact round‑trip

    > `save_cache_artifacts()` → bytes blob → `load_cache_artifacts(...)` hot‑loads functions.
  * PGO swap & generic caches

    > Registers arbitrary cache artifacts via `CacheArtifactFactory.register`. Shows the generic hot‑load API surface.
  * Guard behavior in cache lookups

    > `test_cache_guard` demonstrates guard divergence leading to a fresh compile.

**LLM optimization ideas**

* Capture and ship **pre‑tuned**, **bucket‑specialized** artifacts (kernels + PGO profiles) for your top SKUs; verify parity with hot‑load tests.

---

## 3) Other notable test families (brief)

* **Compiled autograd**

  > `test_compiled_autograd.py` focuses on compiled backward partitions and fake‑tensor mode correctness; CI often flags verbosity/graph mismatch failures.
* **AOTInductor**

  > `test_aot_inductor.py` checks AOT ABI and model export; known gaps with ops like `torch.select` documented with a proposed test.
* **Control flow**

  > `test_control_flow.py` targets Dynamo’s graph‑break/capture coverage for branches/loops; occasionally trips platform decorators.

---

## 4) API surface & entrypoints (documented and semi‑undocumented)

* **Dynamo/compile**: `torch.compile`, `torch.compiler.set_stance`, `torch._dynamo.reset`, `TORCH_LOGS=recompiles` for diagnostics. Docs: dynamic shapes.
* **Inductor config**: `torch._inductor.config.patch({...})` toggles IR/debug/autotune.
* **FX post‑grad logs**: `logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")` is the sanctioned hook to inspect FX after functionalization.
* **Codegen inspection**: `run_and_get_triton_code`, `run_and_get_kernels`, `run_and_get_cpp_code`.
* **Caching**: `torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()` + counters in `torch._inductor.counters`.

---

## 5) vLLM vs upstream tests (short contrast)

* vLLM adds **product‑shaped** tests (piecewise capture, attention/quant fusions, TP/SP) and a **Dynamo artifact hot‑start** test; upstream focuses on **platform‑shaped** coverage (opinfo, autotune backends, compiled autograd, AOT ABI).

---

## 6) Practical recipes (how to replicate these checks yourself)

* **Assert no recompiles**: wrap micro‑model in `CompileCounterWithBackend` and set stance; verify `frame_count`/counters.
* **Inspect post‑grad FX**: replicate `get_post_grad_graph`; assert your pass injected/folded the intended pattern.
* **Check fusion/codegen**: record Triton source and grep for expected kernels or tile configs.
* **Cache round‑trip**: use `save_cache_artifacts`/`load_cache_artifacts` to pre‑seed warm starts.

---

### Appendix A — small code anchors

* Post‑grad log capture:

  > `log_stream, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")`
* Test harness config patch:

  > `config.patch({"debug": True, "debug_index_asserts": True, ...})`
* Cache artifacts round‑trip:

  > `artifacts = torch.compiler.save_cache_artifacts(); load_cache_artifacts(artifact_bytes)`
* Compile counter backend:

  > `backend = CompileCounterWithBackend("inductor"); @torch.compile(backend=backend)`

---

> Questions to investigate next: map **specific tests** (e.g., `test_max_autotune_decompose_k_dynamic_input`) to the Triton configs they generate and identify which KV‑friendly tile heuristics reduce decode P50/P99.

---

# 7) Per-test call traces (PyTorch `test/inductor`)

> These traces follow the *actual* `torch.compile` → Inductor pipeline. I list the knobs each test toggles, the order of callbacks, and what is asserted. Links point to the upstream tests or issues that show the behavior when direct linking to files is brittle.

## 7.1 `test_max_autotune.py` — Max‑autotune, template caches, and backend selection

**Purpose.** Stress the **autotune** path, Triton/CUTLASS backend choice, and template/kernel caching. It also checks remote cache stats and correctness for tricky shapes (e.g., non‑contiguous mm, small‑K GEMMs, conv→cat fusions).

**Common harness**

```py
# Typical knobs these tests toggle
import torch, torch._inductor.config as cfg
cfg.max_autotune = True  # or torch.compile(mode="max-autotune")
cfg.triton.store_cubin = True  # enable template code caching
```

**Representative tests & what they hit**

### A) `TestMaxAutotune::test_non_contiguous_input_mm_plus_mm`

* **Entry**: `@torch.compile(backend="inductor")` → Dynamo capture → guards on shapes/strides.
* **Functionalization/decomp**: `mm` + `view`/`transpose` lifted to Prim ops; non‑contiguous inputs enforce extra copies if needed.
* **Inductor IR & fusion**: scheduler tries to fuse `mm + mm` chain; if stride patterns differ, creates separate Triton kernels.
* **Autotune**: tries multiple tilings (TF32 may be enabled) then **picks best**; test asserts numerical closeness vs eager.
* **Why this test exists**: exposes precision/tile bugs on newer GPUs (H100/B200) and TF32 corner cases.
* **Refs**: failure snapshots show assertion and line anchors in this file (e.g., around the mm+mm assertion). Link examples:

  * H100 failures summary: [https://github.com/pytorch/pytorch/issues/160305](https://github.com/pytorch/pytorch/issues/160305)
  * Disabled/flaky history: [https://github.com/pytorch/pytorch/issues/126867](https://github.com/pytorch/pytorch/issues/126867) , [https://hud.pytorch.org/disabled](https://hud.pytorch.org/disabled)

### B) `TestMaxAutotune::test_mm_k_1`

* **Focus**: K=1 edge case pushes template chooser to odd tiles; confirms Inductor doesn’t over‑specialize or generate scalarized slow paths.
* **Trace**: Dynamo → post‑grad → Inductor IR → Triton kernel selection → autotune search (very small space) → correctness.
* **Refs**: disable notice and tracking: [https://github.com/pytorch/pytorch/issues/159000](https://github.com/pytorch/pytorch/issues/159000)

### C) `TestMaxAutotune::test_conv_cat`

* **Focus**: pattern that fuses convolution followed by `cat` into a single Triton graph (POI kernel naming).
* **Trace**: Dynamo capture → decompose conv → scheduler emits fused Triton with `triton_poi_fused_cat_*` symbol; the test
  **greps generated code** for the kernel name and asserts presence.
* **Refs**: upstream issue with file‑check snippet and failing symbol: [https://github.com/pytorch/pytorch/issues/154229](https://github.com/pytorch/pytorch/issues/154229)

### D) `TestMaxAutotuneRemoteCache::test_max_autotune_remote_caching[_dynamic]`

* **Focus**: counts **remote cache** puts/gets/misses for autotune results to ensure reuse across runs.
* **Trace**: same compile path; at autotune **decision point**, Inductor consults a remote cache; the test compares stats objects.
* **Refs**: example assertion mismatch and the exact class name: [https://github.com/pytorch/pytorch/issues/145361](https://github.com/pytorch/pytorch/issues/145361)

### E) `TestMaxAutotune::test_triton_template_generated_code_caching_mm_plus_mm`

* **Focus**: verifies **template source caching** for Triton kernels across runs (hit/miss counters should match expectations).
* **Trace**: first run populates template cache → subsequent runs check `hits()` / `misses()`; the test asserts expected counts.
* **Refs**: line‑anchored failure (assertEqual on hits): [https://github.com/pytorch/pytorch/issues/157878](https://github.com/pytorch/pytorch/issues/157878)

### F) `TestMaxAutotune::*_search_space_{DEFAULT,EXHAUSTIVE}`

* **Focus**: toggles Inductor’s autotune **search space** breadth; asserts kernel count/accuracy.
* **Refs**: CI HUD shows these parametrizations and failures: [https://www.torch-ci.com/tests/testInfo?file=inductor%2Ftest\_max\_autotune.py\&name=test\_autotune\_conv1x1\_search\_space\_DEFAULT\&suite=TestMaxAutotune](https://www.torch-ci.com/tests/testInfo?file=inductor%2Ftest_max_autotune.py&name=test_autotune_conv1x1_search_space_DEFAULT&suite=TestMaxAutotune)

**Call sequence (max‑autotune tests)**

1. `torch.compile(mode="max-autotune")` (or `cfg.max_autotune=True`).
2. **Dynamo**: bytecode → FX graph + guards.
3. **AOT/functionalization**: PrimTorch decompositions.
4. **Inductor compile\_fx**: schedule & lower; choose **Triton vs CUTLASS**.
5. **Autotune loop**: generate candidates → run microbench → record **best config**; consult **remote cache** if enabled.
6. **Emit & run**: materialize kernels; tests may grep generated code or poll counters.

Useful knobs to mirror in your experiments:

```py
import torch._inductor.config as cfg
cfg.max_autotune = True
cfg.triton.unique_kernel_names = True  # simplifies grepping emitted code
cfg.log.compile_dynamic_shapes = True
```

---

## 7.2 `test_compiled_autograd.py` — Compiled backwards, guard validation, and fake‑tensor mode

**Purpose.** Validate **compiled autograd**’s partitioning and guard logic, and interactions with fake‑tensor/functorch.

**Common harness**

```py
import torch
from torch._dynamo import reset
from torch._inductor import config as cfg
# Some tests toggle verbose logs and guard validation
cfg.debug = True
```

**Representative tests & what they hit**

### A) `TestAutogradWithCompiledAutograd::test_backward_twice_without_saved_values`

* **Entry**: `torch.compile(backend="inductor")` on a module that calls `backward()` twice without retained saved tensors.
* **Trace**: Dynamo capture (fwd) → AOTAutograd partitions fwd/bwd → compiled bwd graph runs → **guard validation** ensures input alias sets and saved tensors expectations hold → triggers duplicate‑tensor guard failure.
* **Refs** (error and test node): [https://github.com/pytorch/pytorch/issues/129938](https://github.com/pytorch/pytorch/issues/129938)

### B) `TestCompiledAutograd::test_mismatch_fake_tensor_mode` (and kin)

* **Focus**: fake‑tensor mode purge between compilations; ensures compiled autograd does not leak real tensors into fakes.
* **Trace**: sets fake‑tensor mode, compiles, executes fwd/bwd; compares logs/counters to assert no mode mismatch.
* **Refs**: CI HUD sample: [https://www.torch-ci.com/tests/testInfo?file=inductor%2Ftest\_compiled\_autograd.py\&name=test\_mismatch\_fake\_tensor\_mode\&suite=TestCompiledAutograd](https://www.torch-ci.com/tests/testInfo?file=inductor%2Ftest_compiled_autograd.py&name=test_mismatch_fake_tensor_mode&suite=TestCompiledAutograd)

**Call sequence (compiled‑autograd tests)**

1. `torch.compile(...)` wraps fwd; AOTAutograd extracts fwd/bwd functions.
2. **Functionalization** ensures mutation‑free graphs for both.
3. **Partitioner** decides cut points (saves vs recompute); emits FX for bwd.
4. **Inductor** lowers both graphs; runtime installs **guard validators** for saved tensors and aliasing.
5. Tests assert guard outcomes and/or compare eager vs compiled grads.

Debug aids:

```py
import torch._logging as tlog
tlog.set_logs(inductor=True, compiled_autograd=True)
```

---

## 7.3 `test_control_flow.py` — `torch.cond`/branches, shapes, and decompositions

**Purpose.** Stress **control‑flow** lowering: `torch.cond`, loops, and dynamic shapes. Confirms AOTAutograd fully decomposes nested branch functions and that Inductor respects shape guards.

**Representative tests & what they hit**

### A) `TestControlFlow::test_cond_control_flow_with_precomputed_size`

* **Entry**: model with `torch.cond` where branch shapes depend on a precomputed size.
* **Trace**: Dynamo capture of cond → AOTAutograd must **fully decompose** inner `true_fn`/`false_fn`; symbol guards on sizes → Inductor lowers chosen branch; test asserts correctness and no surprise recompiles.
* **Refs**: XPU break report for this exact test, links back to the PR that added it: [https://github.com/pytorch/pytorch/issues/130426](https://github.com/pytorch/pytorch/issues/130426)

### B) Nested `torch.cond` decomposition gap (tracked)

* **Context**: AOTAutograd historically missed full decomposition of nested cond bodies, causing graph‑breaks or incorrect FX.
* **Refs**: decomposition bug and exact repro pointing into this test module: [https://github.com/pytorch/pytorch/issues/120160](https://github.com/pytorch/pytorch/issues/120160) , and a symbol‑size trace: [https://github.com/pytorch/pytorch/issues/120160#issuecomment](https://github.com/pytorch/pytorch/issues/120160#issuecomment)

**Call sequence (control‑flow tests)**

1. `torch.compile(...)` with models using `torch.cond`/loops.
2. **Dynamo** emits FX with `cond` nodes; installs guards on shape symbols.
3. **AOTAutograd** extracts branch subgraphs; must decompose nested fns.
4. **Inductor** lowers branch graphs; ensures runtime selection doesn’t break cudagraph/tree policies if enabled.
5. Tests compare outputs and may assert on **no recompiles** when the precomputed size remains bucketed.

---

## 7.4 Quick “how to reproduce the logs and counters”

Use the same hooks the umbrella test harness uses:

```py
import torch
import torch._inductor.config as cfg
from torch.testing._internal.logging_utils import logs_to_string

cfg.debug = True
with logs_to_string("torch._inductor.compile_fx", "post_grad_graphs") as (log, ctx):
    compiled = torch.compile(fn, backend="inductor")
    compiled(*inputs)
print(log.getvalue())  # post-grad FX IR
```

And for caches/counters in autotune tests:

```py
from torch._inductor import counters
# e.g., counters["inductor"]["fxgraph_cache_hit"]
```

---

### Notes

* When direct deep-links to `test_max_autotune.py` are flaky, I referenced CI HUD pages and GitHub issues that quote file names, test case names, and even **line numbers** where asserts fail. Those provide stable anchors for cross‑checking emitted kernels, cache stats, and failure modes across GPU backends.

---

## References (fixed links)

* PyTorch tests: [`test/inductor/test_max_autotune.py`](https://github.com/pytorch/pytorch/blob/main/test/inductor/test_max_autotune.py), [`test/inductor/test_codecache.py`](https://github.com/pytorch/pytorch/blob/main/test/inductor/test_codecache.py), [`test/inductor/test_compiled_autograd.py`](https://github.com/pytorch/pytorch/blob/main/test/inductor/test_compiled_autograd.py), [`test/inductor/test_control_flow.py`](https://github.com/pytorch/pytorch/blob/main/test/inductor/test_control_flow.py), [`test/inductor/test_torchinductor.py`](https://github.com/pytorch/pytorch/blob/main/test/inductor/test_torchinductor.py), [`test/inductor/test_torchinductor_opinfo.py`](https://github.com/pytorch/pytorch/blob/main/test/inductor/test_torchinductor_opinfo.py)
* Issues/threads mentioned: non‑contiguous mm autotune [#159914](https://github.com/pytorch/pytorch/issues/159914); codecache failures [#154250](https://github.com/pytorch/pytorch/issues/154250), flake tracker [#133957](https://github.com/pytorch/pytorch/issues/133957); B200 umbrella [#162178](https://github.com/pytorch/pytorch/issues/162178); compiled‑autograd guard validation [#129938](https://github.com/pytorch/pytorch/issues/129938); nested `torch.cond` decomposition [#120160](https://github.com/pytorch/pytorch/issues/120160); autotune/triton pin rollups [#154206](https://github.com/pytorch/pytorch/issues/154206).
* Docs: [Inductor profiling](https://docs.pytorch.org/docs/stable/torch.compiler_inductor_profiling.html), [compiler troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html), [torch.compile intro](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), [compiled autograd tutorial](https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html)
