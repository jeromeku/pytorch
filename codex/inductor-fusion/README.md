# TorchInductor Fusion — What, Where, and How To Test

Updated: 2025-09-09

This guide explains how the Inductor backend decides what can be fused into a single kernel, where in the torch.compile pipeline those decisions are made, and how to isolate and verify fusion behavior with small runnable snippets. All file references are repo-relative with line anchors for quick navigation.

---

## Where Fusion Happens In The Pipeline

High-level flow (simplified):
- torch.compile → TorchDynamo capture (FX graph + guards)
- AOTAutograd export (fw/bw graphs)
- Inductor post-grad passes (normalization, partitioning, pattern rewrites)
- Inductor IR lowering (buffers, indexing, deps)
- Scheduler builds fusion groups and decides fusing legality and profitability
- Codegen emits kernels for each fusion group

Fusion is primarily decided in the Scheduler and in backend-specific fusion hooks, with additional heuristics in V.choices.

Key code touchpoints
- Scheduler (fusion core):
  - torch/_inductor/scheduler.py:4141–4154 can_fuse orchestrates checks across generic heuristics and backend hooks.
  - torch/_inductor/scheduler.py:4156–4207 can_fuse_vertical checks legality for producer→consumer fusion (vertical).
  - torch/_inductor/scheduler.py:1583–1700 ForeachKernelSchedulerNode.can_fuse handles foreach-style grouping.
  - torch/_inductor/scheduler.py:3648–3720 memory heuristics (fusion_accumulate_large_reads, are_long_distant_nodes, decide_fusion_fail_reason).
- Fusion heuristics (generic, not correctness-critical):
  - torch/_inductor/choices.py:337–404 can_fuse: “no shared data”, max_fusion_size, peak memory, large reads.
- Backend hooks (device-target specifics):
  - torch/_inductor/codegen/cpp.py:4846–4953 can_fuse_horizontal/vertical; outer-loop fusion and template epilogues.
  - Other backends: torch/_inductor/codegen/cuda/cuda_cpp_scheduling.py; rocm, cutedsl hooks analogous.
- Pattern formation before scheduling (FX-level fusions/normalizations):
  - torch/_inductor/fx_passes group_batch_fusion.py, pattern_matcher.py, pre_grad.py (e.g., conv+bn fusions) shape the graph so the scheduler sees fusion-friendly IR.

Logging tap for fusion decisions
- Scheduler uses an artifact logger: torch/_inductor/scheduler.py:79 `fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")`.
- Capture with `logs_to_string("torch._inductor.scheduler", "fusion")` to get detailed “can fuse/cannot fuse” traces.

---

## How The Scheduler Decides “Can We Fuse?”

1) Shared data and dependence orientation
- Producer→Consumer (vertical fusion): If node2 depends on outputs of node1, can_fuse_vertical enforces that reads/writes alias correctly with no incompatible intermediate deps.
  - Entry: torch/_inductor/scheduler.py:4144–4150 (branch where node2 depends on node1)
  - Logic: torch/_inductor/scheduler.py:4156–4207 (remaining deps, MemoryDep matching, ancestor checks).
- Siblings with shared reads (horizontal fusion): If nodes do not depend on each other but share reads, V.choices and backend hooks decide if merging is beneficial and legal.
  - Entry: torch/_inductor/scheduler.py:4151–4154 → `V.choices.can_fuse_horizontal(...)` + backend’s `can_fuse_horizontal`.

2) Generic fusion heuristics (not required for correctness)
- torch/_inductor/choices.py:337–404 can_fuse:
  - “no shared data” short-circuit (lines 352–378).
  - fusion size cap via config.max_fusion_size (lines 380–386).
  - peak-memory guard (lines 388–391) and large-reads guard controlled by config.realize_acc_reads_size_threshold (393–401).
  - memory-score threshold (423–424) checked later in the scheduler pipeline (see scheduler.py:4121).

3) Backend-specific legality and policies
- CPU wrapper backend: torch/_inductor/codegen/cpp.py:4846–4953
  - Disallows horizontal fusion if templates are involved or fusion is too large (4846–4853).
  - Vertical fusion considers template epilogues and reductions (4942–4953).
  - Outer-loop fusion depth logic (4869–4923) ensures compatible loop nests.

4) Foreach fusion special-case
- torch/_inductor/scheduler.py:1583–1700 defines fusion within foreach groups; cannot fuse foreach with reductions (1632–1655), and requires matching arity.

5) Memory/Distance heuristics
- Accumulated reads threshold: scheduler.py:3665–3672
- Long-distance nodes (prevent fusing far-apart nodes to avoid pressure): scheduler.py:3673–3699
- Fusion fail reason diagnostics for indexing mismatches: scheduler.py:3700–3720 and choices.py:355–373

---

## Isolating The Fusion Stage With Logs

Minimal example: capture “fusion” logger while compiling a small graph. Compare when shapes/indexing enable vs disable shared-data fusion.

Snippet A — basic pointwise chain with shared reads
```
import torch
from torch.testing._internal.logging_utils import logs_to_string

def f(x):
    y = x * 2
    z = y + x  # shared read of x, common case for horizontal fusion
    return z.relu()

log, ctx = logs_to_string("torch._inductor.scheduler", "fusion")
with ctx():
    torch.compile(f)(torch.randn(8192))
print(log.getvalue())
```

What you’ll see
- can_fuse attempts for sibling nodes and vertical pairs, with reasons for rejection (e.g., “exceeds max fusion”, “no shared data”, “memory deps did not match”).

Snippet B — force indexing mismatch to defeat fusion
```
import torch
from torch.testing._internal.logging_utils import logs_to_string

def g(x):
    y = x.roll(1, dims=0)  # misaligned read vs write indexing
    z = y + x
    return z

log, ctx = logs_to_string("torch._inductor.scheduler", "fusion")
with ctx():
    torch.compile(g)(torch.randn(8192))
print(log.getvalue())  # expect messages about indexing mismatch / no shared data
```

---

## Verifying Fusion Via Kernel Count

When more ops fuse, the number of emitted kernels typically decreases. Use run_and_get_kernels to count.

```
import torch
from torch._inductor.utils import run_and_get_kernels

def h(x):
    a = x.sin(); b = a * 2; c = b + a; return c.relu()

X = torch.randn(1 << 18)
with torch._inductor.config.patch({"max_fusion_size": 64}):
    _, k1 = run_and_get_kernels(torch.compile(h), X)
with torch._inductor.config.patch({"max_fusion_size": 1}):
    _, k2 = run_and_get_kernels(torch.compile(h), X)
print("kernels fused:", len(k1), " vs limited:", len(k2))
```

Related knobs
- torch/_inductor/config.py:658 max_fusion_size
- torch/_inductor/choices.py:383 checks fusion size

---

## Vertical Fusion With Template Epilogues (Matmul + ReLU)

Backend can fuse an epilogue op into a template kernel (e.g., GEMM + ReLU).

Code paths
- Backend vertical fusion decisions: torch/_inductor/codegen/cpp.py:4942–4953
- Template epilogue checks: `template_fusion_with_epilogues_supported` is consulted inside these paths.

Snippet — GEMM with epilogue candidate
```
import torch
from torch._inductor.utils import run_and_get_kernels

def mm_relu(a, b):
    return (a @ b).relu()

dev = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(1024, 1024, device=dev)
b = torch.randn(1024, 1024, device=dev)

_, kernels = run_and_get_kernels(torch.compile(mm_relu), a, b)
print("num kernels:", len(kernels))  # expect 1 when epilogue fusion is supported/configured
```

Factors that disable epilogue fusion
- Templates not used (backend path chooses ATen), reductions in consumer, or backend policy says no (cpp.py:4942–4953).

---

## Why Fusion Might Fail (Traceable Reasons)

These are emitted by WhyNoFuse and by V.choices diagnostics:
- “no shared data” or “indexing mismatch” (choices.py:352–378; 355–373 for reason rows)
- “exceeds max fusion” (choices.py:383–386)
- “Fusion will increase peak memory” (choices.py:388–391) → see scheduler.py:3648–3663 for memory-overhead heuristic
- “memory deps did not match” or “intermediate nodes between node1 & node2” (scheduler.py:4192–4205)
- Foreach constraints — reductions cannot fuse with foreach (scheduler.py:1632–1655)
- Backend disallows mixing templates or outer-loop depth incompatible (cpp.py:4838–4856, 4869–4934)

---

## Quick Checklist To Isolate Fusion Decisions
- Enable logs: `logs_to_string("torch._inductor.scheduler", "fusion")`.
- Toggle heuristics:
  - `max_fusion_size`, `score_fusion_memory_threshold`, `realize_acc_reads_size_threshold` in torch/_inductor/config.py.
  - `pick_loop_orders` affects shared indexing structure; see torch/_inductor/scheduler.py:2002–2046.
- Count kernels: `run_and_get_kernels` before/after toggles.

---

## Additional Anchors (for deeper reading)
- Fusion entry and orchestration: torch/_inductor/scheduler.py:4141–4154
- Vertical fusion legality: torch/_inductor/scheduler.py:4156–4207
- Heuristics layer: torch/_inductor/choices.py:337–404, 423–424
- Memory heuristics: torch/_inductor/scheduler.py:3648–3720
- Backend hooks (CPU): torch/_inductor/codegen/cpp.py:4846–4953
- Foreach fusion: torch/_inductor/scheduler.py:1583–1700
- Pattern formation (FX): torch/_inductor/fx_passes/group_batch_fusion.py (multiple fusers), pre_grad.py (conv+bn around 435–546)

---

## Snippets In This Folder
- fusion_logs_basic.py — capture fusion decisions during compile
- fusion_kernels_count.py — compare kernel counts under different fusion limits
- fusion_index_mismatch.py — force indexing mismatch to defeat fusion
- fusion_epilogue_mm.py — GEMM+ReLU epilogue fusion probe

