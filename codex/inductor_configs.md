# TorchInductor Config Flags — What They Do, Where They Flow, and How To Demo

Updated: 2025-09-09

This guide explains key `torch._inductor.config` flags, where they impact the `torch.compile → AOTAutograd → Inductor` pipeline, and provides runnable snippets that show observable differences. Use `config.patch(...)` to toggle flags per-call.

See scripts in `codex/scripts/inductor_config_demos.py` for ready-to-run examples.

---

## How To Toggle Flags
- Patching: `with torch._inductor.config.patch({"fx_graph_cache": False, "triton.autotune_at_compile_time": True}): ...`
- Nested keys work (e.g., `"triton.cudagraphs"`). The config module supports dot-key patching.

Reference: torch/utils/_config_module.py:633 (patch mechanism); torch/_inductor/config.py:1975 (install_config_module).

---

## Caching Flags

- fx_graph_cache
  - What: Enables the FX-graph-level codegen cache used by Inductor.
  - Definition: torch/_inductor/config.py:93
  - Usage: increments in cache hits/misses inside the FX graph cache.
    - Hit/miss counters: torch/_inductor/codecache.py:1548 (hit), 1587 (miss)
    - Compile-time paths: torch/_inductor/compile_fx.py:861, 872–877 (status logging)
  - Demo: `inductor_config_demos.py::demo_fx_graph_cache()` compares `counters["inductor"]["fxgraph_cache_{hit,miss}"]` across runs.

- force_disable_caches
  - What: Hard-disables Inductor/Dynamo caches even if cache flags are enabled (useful for debugging).
  - Definition: torch/_inductor/config.py:149 (aliases torch.compiler.config.force_disable_caches)
  - Usage: torch/_inductor/compile_fx.py:710, 861, 872–877; codegen and triton bundler respect it.
    - See also: torch/_inductor/triton_bundler.py:119; torch/_inductor/runtime/autotune_cache.py:234–270
  - Demo: `demo_force_disable_caches()` shows that hits stay at 0 and only misses increment.

- autotune_local_cache, autotune_remote_cache, bundled_autotune_remote_cache
  - What: Control Triton/CUTLASS autotune caches (local/remote/bundled in artifacts).
  - Definitions: torch/_inductor/config.py:122, 134, 146
  - Usage: torch/_inductor/runtime/autotune_cache.py:83–86, 352–366, 445; torch/_inductor/codegen/triton.py:3997–4003
  - Demo: `demo_autotune_caches()` compiles a matmul with and without caches, printing whether a bundled cache blob is emitted.

---

## Graph Normalization & Post-Grad Flags

- joint_graph_constant_folding
  - What: Enables constant folding on the joint (pre/post-grad) graph.
  - Definition: torch/_inductor/config.py:722
  - Usage: torch/_inductor/fx_passes/joint_graph.py:586
  - Demo: `demo_joint_graph_constant_folding()` uses post-grad logs to show op-count differences with folding on/off.

---

## IR, Fusion, and Scheduling Flags

- max_fusion_size
  - What: Upper bound on number of nodes allowed into one fusion group.
  - Definition: torch/_inductor/config.py:658
  - Usage: torch/_inductor/choices.py:383
  - Demo: `demo_max_fusion_size()` counts emitted kernels via `run_and_get_kernels` as `max_fusion_size` is lowered.

- unroll_reductions_threshold
  - What: Size threshold to replace small reductions with pointwise code (affects lowering/codegen shapes).
  - Definition: torch/_inductor/config.py:670
  - Usage: torch/_inductor/ir.py:1559; torch/_inductor/lowering.py:6017, 7193
  - Demo: `demo_unroll_reductions_threshold()` prints whether reduction constructs appear in emitted code.

- pick_loop_orders
  - What: Allow scheduler to re-order loops for better locality/perf.
  - Definition: torch/_inductor/config.py:217
  - Usage: torch/_inductor/scheduler.py:2043
  - Demo: `demo_pick_loop_orders()` runs a small stencil; compare loop headers in emitted code.

---

## Codegen & Autotune Flags

- triton.autotune_at_compile_time
  - What: Perform autotune during compile (one-pass codegen); affects wrapper behavior and when tuning happens.
  - Definition: torch/_inductor/config.py:1310
  - Usage: torch/_inductor/compile_fx.py:2068–2075; torch/_inductor/codegen/wrapper.py (multiple call sites)
  - Demo: `demo_autotune_at_compile_time()` emits code twice and shows structural differences (e.g., wrapper emits tuned kernel directly).

- max_autotune, max_autotune_pointwise, max_autotune_gemm_backends
  - What: Enable slow autotuning passes and control backend pools for GEMM/CONV.
  - Definitions: torch/_inductor/config.py:438–444, 492–502
  - Usage: torch/_inductor/runtime/triton_heuristics.py:2468–2669, select_algorithm & kernel templates
  - Demo: `demo_max_autotune_backends()` toggles backends and prints which backend path the produced code chose.

- triton.use_block_ptr
  - What: Prefers Triton block-pointer path; constrains index dtypes (32-bit) and enables certain templates.
  - Definition: torch/_inductor/config.py:1395
  - Usage: torch/_inductor/codegen/triton_utils.py:134–147; torch/_inductor/codegen/triton.py:2114, 2304
  - Demo: `demo_use_block_ptr()` greps the emitted code for `tl.int32` vs `tl.int64` size arg signatures.

---

## Runtime Flags

- triton.cudagraphs
  - What: Enable CUDA Graphs (and cudagraph trees) in the emitted runtime.
  - Definition: torch/_inductor/config.py:1215
  - Usage: torch/_inductor/compile_fx.py:831–858, 2062–2170; torch/_inductor/output_code.py:198–216, 655–702; cudagraph utils
  - Demo: `demo_cudagraphs()` runs a function twice; with cudagraphs on, you should observe replay path in logs and stable allocations.

- size_asserts
  - What: Emit size/stride assertions in Python wrapper to guard shapes/strides.
  - Definition: torch/_inductor/config.py:206
  - Usage: torch/_inductor/ir.py:6265–6274; torch/_inductor/codegen/wrapper.py:1300–1410
  - Demo: `demo_size_asserts()` greps emitted wrapper for `assert_size_stride(...)` on/off.

- fallback_random and implicit_fallbacks
  - What: Allow runtime fallback for random ops (and implicit fallbacks for ops lacking lowerings).
  - Definitions: torch/_inductor/config.py:618 (fallback_random), 621 (implicit_fallbacks)
  - Usage: torch/_inductor/lowering.py:2065–2176 (random), torch/_inductor/graph.py:1198–1206 (implicit fallbacks)
  - Demo: `demo_fallback_random()` compiles a bernoulli/dropout example with `fallback_random=True` vs `False`.

---

## Usage Snippets (inline)

Below are condensed examples; the full runnable versions with diffs and logging live in `codex/scripts/inductor_config_demos.py`.

1) FX graph cache vs disabled
```
import torch
from torch._dynamo.utils import counters

def f(x):
    return (x.sin() + x.cos()).relu()

x = torch.randn(8, device="cuda")

with torch._inductor.config.patch({"fx_graph_cache": True}):
    cnt_before = counters["inductor"]["fxgraph_cache_hit"]
    g = torch.compile(f)
    g(x); g(x)
    print("hits+=", counters["inductor"]["fxgraph_cache_hit"] - cnt_before)

with torch._inductor.config.patch({"fx_graph_cache": False}):
    counters.clear()
    g = torch.compile(f)
    g(x); g(x)
    print("hits when disabled:", counters["inductor"]["fxgraph_cache_hit"])  # expect 0
```

2) Fusion size (kernel count)
```
import torch
from torch._inductor.utils import run_and_get_kernels

def f(x):
    y = x.sin(); z = y * 2 + y.cos(); return z.relu()

with torch._inductor.config.patch({"max_fusion_size": 64}):
    _, kernels1 = run_and_get_kernels(torch.compile(f), torch.randn(1024, device="cuda"))
with torch._inductor.config.patch({"max_fusion_size": 1}):
    _, kernels2 = run_and_get_kernels(torch.compile(f), torch.randn(1024, device="cuda"))
print(len(kernels1), "kernels vs", len(kernels2), "kernels")  # expect more with size=1
```

3) Reduction unrolling threshold
```
import torch
from torch._inductor.utils import run_and_get_triton_code

def reduce_small(x): return x.sum(dim=-1)
x = torch.randn(32, 4, device="cuda")  # small last-dim reduction

with torch._inductor.config.patch({"unroll_reductions_threshold": 1}):
    code_small = run_and_get_triton_code(torch.compile(reduce_small), x)
with torch._inductor.config.patch({"unroll_reductions_threshold": 128}):
    code_large = run_and_get_triton_code(torch.compile(reduce_small), x)
print("has 'reduce' in code small?", "reduce" in code_small)
print("has 'reduce' in code large?", "reduce" in code_large)
```

4) Use block pointers (index dtype)
```
import torch
from torch._inductor.utils import run_and_get_triton_code

def f(a, b): return (a @ b).relu()
a = torch.randn(128, 64, device="cuda"); b = torch.randn(64, 128, device="cuda")

with torch._inductor.config.patch({"triton.use_block_ptr": False}):
    code0 = run_and_get_triton_code(torch.compile(f), a, b)
with torch._inductor.config.patch({"triton.use_block_ptr": True}):
    code1 = run_and_get_triton_code(torch.compile(f), a, b)
print("int64 meta present?", "tl.int64" in code0, "→ after True:", "tl.int64" in code1)
```

5) Size asserts in wrapper
```
import torch
from torch._inductor.utils import run_and_get_triton_code

def f(x): return x.view(x.shape)
x = torch.randn(32, 32, device="cuda")

with torch._inductor.config.patch({"size_asserts": True}):
    code_on = run_and_get_triton_code(torch.compile(f), x)
with torch._inductor.config.patch({"size_asserts": False}):
    code_off = run_and_get_triton_code(torch.compile(f), x)
print("assert_size_stride present?", "assert_size_stride" in code_on, "off?", "assert_size_stride" in code_off)
```

6) Compile-time Triton autotune
```
import torch
from torch._inductor.utils import run_and_get_triton_code

def f(x): return (x * x).relu()
x = torch.randn(1 << 20, device="cuda")

with torch._inductor.config.patch({"triton.autotune_at_compile_time": False}):
    code_defer = run_and_get_triton_code(torch.compile(f), x)
with torch._inductor.config.patch({"triton.autotune_at_compile_time": True}):
    code_ct = run_and_get_triton_code(torch.compile(f), x)
print("len(code_defer)", len(code_defer), "len(code_ct)", len(code_ct))
```

7) CUDA Graphs
```
import torch
def f(x): return (x.sin() + 1.0).cos()
x = torch.randn(8192, device="cuda")

with torch._inductor.config.patch({"triton.cudagraphs": True}):
    g = torch.compile(f)
    g(x); g(x)  # second run should replay

with torch._inductor.config.patch({"triton.cudagraphs": False}):
    g2 = torch.compile(f)
    g2(x); g2(x)
```

---

## Full Demo Script
- See: `codex/scripts/inductor_config_demos.py`
- Provides per-flag demos that print short, grep-friendly evidence of changed artifacts.

