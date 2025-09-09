# TorchInductor Tests — Detailed Stage Mapping and Isolated Runners

Purpose: map test/inductor coverage to the major torch.compile → AOTAutograd → Inductor pipeline stages, with concrete example tests (path:line) and ready-to-run commands to exercise each component in isolation.

Updated: 2025-09-09

---

## How To Run One Test In Isolation
- Use pytest’s node ids to run a single test or class.
- Optional: set `TORCH_LOGS`, `PYTORCH_TEST_WITH_SLOW`, or Inductor/Dynamo config flags.

Examples
- Run one test function: `pytest -q test/inductor/test_codecache.py::TestFxGraphCache::test_cache_guard -q`
- Filter by pattern: `pytest -q test/inductor/test_max_autotune.py -k decompose_k_dynamic_input`
- Enable post-grad FX logs: `TORCH_LOGS=post_grad_graphs pytest -q test/inductor/test_torchinductor.py::TestCase::test_bool`

See `codex/scripts/inductor_component_tests.sh` for scripted shortcuts.

---

## Stage 2.1 — Dynamo Capture & Guards
What this stage does
- Intercepts Python frames and builds an FX graph (TorchDynamo), recording guards about shapes/dtypes/devices to decide reuse vs recompile.
- Good coverage checks: no accidental graph breaks, minimal recompiles, correct handling of dynamic shapes, robust control-flow capture.

Where these tests fit in the pipeline
- Entry point: `torch.compile` → Dynamo trace → AOTAutograd export → Inductor compile.
- Guard/capture policy directly influences cache keys and recompile frequency downstream.

Representative tests and themes
- test/inductor/test_control_flow.py
  - test_cond_simple_control_flow:324 exercises if/else capture and branch guards (device/dtype/shape), ensuring no spurious breaks.
  - test_cond_simple_with_int_closure:338 checks closure-captured scalars still allow a single compiled graph.
  - test_cond_control_flow_with_precomputed_size:366 ensures size-dependent branches are guarded precisely under CUDA requirements.
- test/inductor/test_torchinductor.py
  - setUp/tearDown resets: 341, 342, 348 call `torch._dynamo.reset()` and `torch._inductor.metrics.reset()` to isolate runs between tests.
  - check_model helper: 418 normalizes eager vs compiled parity and asserts numerics/strides/gradients.
- test/inductor/test_torchinductor_dynamic_shapes.py
  - test_constant_fold_uniform_value_dynamic:182 validates shape reasoning under dynamic input, avoiding unnecessary recompiles.
  - test_arange_dynamic:227 probes dynamic loop bounds and symbol handling.

Helpful source anchors
- Reset helpers: `test/inductor/test_torchinductor.py:341`, `:342`, `:348`; `torch/_dynamo/__init__.py:112` (reset definition).
- FX post-grad logs: `torch/testing/_internal/logging_utils.py:192` (`logs_to_string`).

Minimal runnable snippets (no test suite required)
1) Verify capture and recompiles via counters
```
import torch
from torch._dynamo.utils import counters

def branchy(x, t):
    return x.sin() if t > 0 else x.cos()

x = torch.randn(8, device=("cuda" if torch.cuda.is_available() else "cpu"))
counters.clear()
g = torch.compile(branchy)
g(x, 1)
hit_before = counters["inductor"].get("fxgraph_cache_hit", 0)
g(x, 1)  # same guards
print("cache hits +=", counters["inductor"].get("fxgraph_cache_hit", 0) - hit_before)

# New guard (different branch)
g(x, -1)
print("cache misses:", counters["inductor"].get("fxgraph_cache_miss", 0))
```

2) Inspect post‑grad FX graph after capture
```
import torch
from torch.testing._internal.logging_utils import logs_to_string

def f(x):
    y = x.view(x.shape)
    return (y + 1).relu()

log_stream, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
with ctx():
    torch.compile(f)(torch.randn(4))

print(log_stream.getvalue())
```

---

## Stage 2.2 — Functionalization & Decompositions
What this stage does
- Removes mutation by converting in‑place ops to functional form (functionalization), enabling clean AOT export.
- Applies decompositions (PrimTorch/Inductor) to lower complex ops into fusion‑friendly primitives (e.g., mm → blocks + epilogues).

Representative tests and themes
- test/inductor/test_auto_functionalize.py
  - test_auto_functionalize_can_with_default:20 checks that in-place semantics are properly rewritten.
  - test_auto_functionalize_can_with_none_return:43 verifies functionalization under odd returns.
- test/inductor/test_decompose_mem_bound_mm.py
  - test_decompose_bmm:105 (CUDA) and :142 (CPU) ensure mem‑bound bmm decompositions route to intended kernels.
- test/inductor/test_pad_mm.py
  - test_pad_mm_dyn_m:31 and test_pad_mm_dyn_n:101 validate the `pad_mm` pass for dynamic M/N with padding before GEMM.
- test/inductor/test_binary_folding.py
  - test_conv_binary_folding:41 and test_conv_bn_folding:158 confirm folding patterns pre‑fusion.

Helpful source anchors
- pad_mm implementation: `torch/_inductor/fx_passes/pad_mm.py:761` (pad_mm), invoked from post‑grad pipeline.

Minimal runnable snippets
1) Show decomposition effect in post‑grad graph
```
import torch
from torch.testing._internal.logging_utils import logs_to_string

def mm_chain(a, b):
    return (a @ b).relu()

a = torch.randn(32, 64)
b = torch.randn(64, 32)

ls, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
with ctx():
    torch.compile(mm_chain)(a, b)
print(ls.getvalue())  # look for normalized/view/reshape, fused epilogue patterns
```

2) Functionalization of in‑place ops
```
import torch

def inplace(x):
    y = x.clone()
    y.add_(1)
    return y

compiled = torch.compile(inplace)
print(compiled(torch.randn(4)))  # Should functionalize add_
```

---

## Stage 2.3 — FX Post‑Grad Shaping & Partitioning
What this stage does
- After AOTAutograd, Inductor runs post‑grad passes to canonicalize patterns (split/cat, view/reshape), fold constants, and optionally partition graphs.
- Pattern matchers ensure expected shapes/patterns before scheduling.

Representative tests and themes
- test/inductor/test_custom_post_grad_passes.py
  - test_custom_joint_pass_pre:147 / test_custom_joint_pass_post:159 show how pre/post passes modify graphs deterministically.
- test/inductor/test_custom_partitioner_fn.py
  - test_custom_partitioner_fn:30 injects a custom partitioning function to isolate parts of the graph.
- test/inductor/test_split_cat_fx_passes.py
  - test_split_normalization:36 and test_consecutive_split_merge:146 verify split→cat normalization to preferred idioms.
- test/inductor/test_split_cat_fx_aten_passes.py
  - test_split_cat_post_grad:259 and test_select_cat_post_grad:328 ensure ATen‑side normalization aligns with post‑grad rules.
- test/inductor/test_pattern_matcher.py
  - test_mm_plus_mm:88 / test_fused_int_mm_mul:151 confirm matcher rules target intended fusion shapes.

Minimal runnable snippets
1) Observe split/cat normalization in logs
```
import torch
from torch.testing._internal.logging_utils import logs_to_string

def split_cat(x):
    a, b = torch.tensor_split(x, 2, dim=-1)
    return torch.cat([a, b], dim=-1)

ls, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
with ctx():
    torch.compile(split_cat)(torch.randn(4, 8))
print(ls.getvalue())
```

2) Custom partitioner sketch (conceptual)
```
# Shows how a custom partitioner could be wired; consult
# test/inductor/test_custom_partitioner_fn.py:30 for a working example.
import torch
from torch._inductor.custom_graph_pass import CustomPartitionerFn

class MyPartitioner(CustomPartitionerFn):
    def __call__(self, gm, inputs):  # decide partitioning
        return [(gm.graph.nodes, {})]

with torch._inductor.config.patch({"custom_graph_passes": []}):
    # apply via compile config hook (see compile_fx)
    pass
```

---

## Stage 2.4 — Inductor IR & Fusion
What this stage does
- Lowers the FX graph into Inductor’s Loop IR, decides legal fusion groups, schedules loops, and applies tiling/ordering transforms.

Representative tests and themes
- test/inductor/test_inductor_scheduler.py — scheduler legality and transform decisions across devices.
- test/inductor/test_loop_ordering.py — class TestTiling:955 covers tiling heuristics and ordering profitability.
- test/inductor/test_kernel_optimization.py — class TestKernelOptimization:31 validates kernel‑level simplifications.
- test/inductor/test_segmented_tree.py — test_basic_construction:46, test_max_query_matches_naive:57 exercise segmented tree data structures that back fusion decisions.
- test/inductor/test_group_batch_fusion.py — attention/group‑batch patterns and tradeoffs.

Minimal runnable snippets
1) Fusion group size impact (kernel count)
```
import torch
from torch._inductor.utils import run_and_get_kernels

def f(x):
    y = x.sin(); z = y * 2 + y.cos(); return z.relu()

X = torch.randn(1 << 16)
with torch._inductor.config.patch({"max_fusion_size": 64}):
    _, k1 = run_and_get_kernels(torch.compile(f), X)
with torch._inductor.config.patch({"max_fusion_size": 1}):
    _, k2 = run_and_get_kernels(torch.compile(f), X)
print("kernels:", len(k1), "→", len(k2))
```

2) Reduction unrolling threshold
```
import torch
from torch._inductor.utils import run_and_get_triton_code

def reduce_small(x): return x.sum(dim=-1)
X = torch.randn(32, 4)
with torch._inductor.config.patch({"unroll_reductions_threshold": 1}):
    c1 = run_and_get_triton_code(torch.compile(reduce_small), X)
with torch._inductor.config.patch({"unroll_reductions_threshold": 128}):
    c2 = run_and_get_triton_code(torch.compile(reduce_small), X)
print("contains 'reduce' op?", ("reduce" in c1), ("reduce" in c2))
```

---

## Stage 2.5 — Codegen & Autotune (Triton/CUTLASS)
What this stage does
- Emits backend kernels (Triton/CUTLASS/CPP), and optionally autotunes them (compile‑time or runtime) for performance.

Representative tests and themes
- test/inductor/test_triton_kernels.py
  - test_triton_kernel_with_kernel_param:100 ensures kernel parameter plumbing behaves as expected.
  - test_triton_kernel_functionalize:167 validates functionalization around Triton kernels.
- test/inductor/test_triton_heuristics.py
  - test_triton_config:82 and test_artificial_zgrid:132 probe grid/config heuristics.
- test/inductor/test_cutlass_backend.py
  - test_check_paths:203 validates CUTLASS integration paths; test_max_autotune_cutlass_threshold:218 gates autotune intensity.
- test/inductor/test_max_autotune.py
  - test_max_autotune_decompose_k_dynamic_input:1165 targets split‑K autotune; suites cover precompile/subproc/remote cache modes.
- test/inductor/test_best_config.py — best‑config cache key includes Triton metadata:52.
- test/inductor/test_codegen_triton.py — test_config_of_sizearg:35 covers size‑arg typing and signatures.

Minimal runnable snippets
1) Autotune at compile time vs deferred
```
import torch
from torch._inductor.utils import run_and_get_triton_code

def pointwise(x): return (x * x).relu()
X = torch.randn(1<<20)
with torch._inductor.config.patch({"triton.autotune_at_compile_time": False}):
    c0 = run_and_get_triton_code(torch.compile(pointwise), X)
with torch._inductor.config.patch({"triton.autotune_at_compile_time": True}):
    c1 = run_and_get_triton_code(torch.compile(pointwise), X)
print(len(c0), len(c1))
```

2) Block‑pointer favor toggles index dtype (tl.int64 vs tl.int32)
```
import torch
from torch._inductor.utils import run_and_get_triton_code

def gemm(a, b): return (a @ b).relu()
a = torch.randn(128, 64); b = torch.randn(64, 128)
with torch._inductor.config.patch({"triton.use_block_ptr": False}):
    c0 = run_and_get_triton_code(torch.compile(gemm), a, b)
with torch._inductor.config.patch({"triton.use_block_ptr": True}):
    c1 = run_and_get_triton_code(torch.compile(gemm), a, b)
print("tl.int64 present?", "tl.int64" in c0, "→", "tl.int64" in c1)
```

---

## Stage 2.6 — Runtime: CUDA Graphs & Collectives
What this stage does
- Applies cudagraph recording/replay (optionally trees) and runtime policies for stable buffers/streams.

Representative tests and themes
- test/inductor/test_cudagraph_trees.py
  - test_run_simple:209 and test_rng_trees:239 validate replay and RNG‑safe trees.
- test/inductor/test_static_cuda_launcher.py
  - test_basic:68, test_signed_integers:119 ensure static launcher correctness.
- test/inductor/test_memory_planning.py — buffer lifetimes and reuse.
- test/inductor/test_device_assert.py — device asserts surface at the correct time (20, 40).

Minimal runnable snippet
```
import torch

def f(x): return (x.sin() + 1.0).cos()
X = torch.randn(8192)
with torch._inductor.config.patch({"triton.cudagraphs": True}):
    g = torch.compile(f)
    g(X); g(X)  # second call should replay
print("cudagraphs replayed (no error)")
```

---

## Stage 2.7 — Caching & Hot‑Load
What this stage does
- Controls FX‑graph cache reuse, code artifact serialization, autotune cache (local/remote/bundled), and subprocess compile modes.

Representative tests and themes
- test/inductor/test_codecache.py
  - test_cache_load_function:172 round‑trips a function through cache; test_cache_guard:1447 demonstrates guard‑based diverge leading to recompiles.
  - TestAutotuneCache, TestStandaloneCompile suites cover autotune cache and standalone compile artifacts.
- test/inductor/test_remote_cache.py — remote cache logging (45, 60).
- test/inductor/test_cudacodecache.py — CUDA code cache load/error (41, 69).
- test/inductor/test_compile_subprocess.py — progressive/async subproc compile (100, 178).

Minimal runnable snippets
1) FX‑graph cache hits/misses
```
import torch
from torch._dynamo.utils import counters

def f(x, s):
    return x * s

x = torch.randn(8)
counters.clear()
g = torch.compile(f, fullgraph=True)
g(x, 2); g(x, 2)
print("hits:", counters["inductor"]["fxgraph_cache_hit"], "misses:", counters["inductor"]["fxgraph_cache_miss"])

g(x, 3)  # new guard path
print("after shape/const change → misses:", counters["inductor"]["fxgraph_cache_miss"])
```

2) Artifact hot‑load
```
import torch

def f(x): return (x + 1).relu()
g = torch.compile(f)
g(torch.randn(8))
art = torch.compiler.save_cache_artifacts()
torch.compiler.load_cache_artifacts(art[0])  # should rehydrate cache entries
print("loaded artifacts")
```

---

## Other Notable Families (Cross-Cutting)
- Opinfo and parity
  - test/inductor/test_torchinductor_opinfo.py::test_comprehensive:1153
  - test/inductor/test_op_completeness.py (module)
- Quantization and FP8
  - test/inductor/test_quantization.py::test_activation_quantization_aten_with_scaling:128
  - test/inductor/test_fp8.py (module)
- AOTInductor ABI and export
  - test/inductor/test_aot_inductor.py::test_simple:172
  - test/inductor/test_aot_inductor_package.py::test_add:212
- Performance/metrics
  - test/inductor/test_perf.py::test_pointwise:102
  - test/inductor/test_metrics.py (module)

---

## Suggested Alternate Grouping
When you’re triaging or adding coverage, grouping by “where fixes land” can be handier than the strict pipeline view:

- Capture & Guards
  - control_flow, dynamic_shapes, stance, counters
  - Files: test_control_flow.py, test_torchinductor_dynamic_shapes.py, test_torchinductor_codegen_dynamic_shapes.py
- Graph Normalization
  - functionalization, decompositions, canonicalizations
  - Files: test_auto_functionalize.py, test_decompose_mem_bound_mm.py, test_mmdecomp.py, test_pad_mm.py, test_binary_folding.py, test_inplacing_pass.py
- Post-Grad Shaping
  - custom passes/partitioning/pattern rewriting
  - Files: test_custom_post_grad_passes.py, test_custom_partitioner_fn.py, test_split_cat_fx_passes.py, test_split_cat_fx_aten_passes.py, test_pattern_matcher.py, test_mkldnn_pattern_matcher.py
- IR & Scheduler
  - loop ordering, segmented trees, kernel-level legality
  - Files: test_inductor_scheduler.py, test_loop_ordering.py, test_kernel_optimization.py, test_segmented_tree.py
- Codegen & Backends
  - Triton/CUTLASS/backends, autotune, templates
  - Files: test_triton_kernels.py, test_triton_heuristics.py, test_cutlass_backend.py, test_max_autotune.py, test_codegen_triton.py, test_combo_kernels.py, test_best_config.py
- Runtime & Launch
  - cudagraph trees, static launcher, memory planning
  - Files: test_cudagraph_trees.py, test_static_cuda_launcher.py, test_memory_planning.py, test_device_assert.py
- Caching & Artifacts
  - FXGraph cache, CUDA code cache, remote caches, subprocess
  - Files: test_codecache.py, test_remote_cache.py, test_cudacodecache.py, test_compile_subprocess.py
- Cross-Cutting
  - opinfo/parity, quant/FP8, AOT, metrics/profiling
  - Files: test_torchinductor_opinfo.py, test_op_completeness.py, test_quantization.py, test_fp8.py, test_aot_inductor*.py, test_perf.py, test_metrics.py, test_profiler.py

This grouping tends to mirror how fixes are deployed (Dynamo vs FX passes vs IR vs codegen vs runtime).

---

## Quick Scripts (Isolation Recipes)

These one-liners run focused tests and enable useful logs.

- Capture & guards
  - `pytest -q test/inductor/test_control_flow.py::test_cond_simple_control_flow`
  - `TORCH_LOGS=recompiles pytest -q test/inductor/test_torchinductor_dynamic_shapes.py::test_constant_fold_uniform_value_dynamic`

- Graph normalization
  - `pytest -q test/inductor/test_auto_functionalize.py::test_auto_functionalize_can_with_default`
  - `pytest -q test/inductor/test_decompose_mem_bound_mm.py::test_decompose_bmm`

- Post-grad shaping
  - `TORCH_LOGS=post_grad_graphs pytest -q test/inductor/test_custom_post_grad_passes.py::test_custom_joint_pass_post`
  - `pytest -q test/inductor/test_split_cat_fx_passes.py::test_split_normalization`

- IR & fusion
  - `pytest -q test/inductor/test_loop_ordering.py::TestTiling`
  - `pytest -q test/inductor/test_segmented_tree.py::test_basic_construction`

- Codegen & autotune
  - `pytest -q test/inductor/test_triton_kernels.py::test_triton_kernel_with_kernel_param`
  - `pytest -q test/inductor/test_max_autotune.py::TestMaxAutotune::test_max_autotune_decompose_k_dynamic_input`

- Runtime
  - `pytest -q test/inductor/test_cudagraph_trees.py::test_run_simple`
  - `pytest -q test/inductor/test_static_cuda_launcher.py::test_basic`

- Caching
  - `pytest -q test/inductor/test_codecache.py::TestFxGraphCache::test_cache_guard`
  - `pytest -q test/inductor/test_cudacodecache.py::test_cuda_load`

For convenience, use the helper script: `codex/scripts/inductor_component_tests.sh`.
