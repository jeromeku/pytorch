# TorchInductor Architecture

This document describes the architecture of `torch._inductor` with a focus on the CUDA/GPU path (Triton and CUTLASS). It follows the framing suggested in https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html: which problems are solved, how components fit together, what invariants hold, how to navigate the code, and how to observe internals.

The goal is to help you: (1) understand where TorchDynamo hands off to Inductor, (2) see how AOTAutograd/functorch participates, (3) follow the sequence of IRs/passes and files, and (4) peek inside each stage, including an end‑to‑end, internal‑API driven trace from `torch.compile` to final codegen.


## Big Picture

- TorchDynamo captures Python programs into an FX graph (ATen ops). It selects a backend. For `backend="inductor"`, it calls Inductor.
- Inductor orchestrates: pre‑grad FX passes → AOTAutograd to form a joint graph + partition → post‑grad FX passes on fwd/bwd → lower FX to Inductor IR → schedule/fuse → codegen (Triton or CUDA templates/CUTLASS) → wrap/launch.
- Most GPU kernels are generated via Triton; GEMMs/convs may use CUDA C++ templates (e.g., CUTLASS) under autotune.


## Entry Points

- Dynamo → Inductor handoff: `torch/_dynamo/backends/inductor.py:inductor()` lazily imports and returns `torch._inductor.compile_fx.compile_fx`.
- Inductor’s public entry for a single FX graph: `torch/_inductor/compile_fx.py:compile_fx()`.
  - Orchestrates end‑to‑end: pre‑grad passes, AOTAutograd, partitioning, post‑grad passes, and then calls the inner compiler.
- Inductor’s per‑graph compiler: `compile_fx_inner()` → `_compile_fx_inner()`.
  - Builds `GraphLowering` (Inductor IR), runs scheduling/fusion, and emits code (Triton/CUDA/etc.).


## AOTAutograd / functorch’s Role

- Inductor calls functorch AOTAutograd to:
  - Build a joint forward+backward FX graph with decompositions, then
  - Partition it (default: `min_cut_rematerialization_partition`) into forward/backward graphs, and
  - Invoke Inductor compilers for each partition.
- Where: `torch/_inductor/compile_fx.py:_compile_fx_main()` creates `fw_compiler`/`bw_compiler` via `compile_fx_forward()` / `compile_fx_backward()` and calls `.._dynamo.backends.common.aot_autograd()`.
- Joint‑graph passes: `torch/_inductor/fx_passes/joint_graph.py:joint_graph_passes()` (constant folding, cleanup, attention/SDPA patterns, etc.).


## IRs and Pass Pipelines (with files)

Below is the sequence of IRs and the passes that transform them. The same sequence runs for forward and backward (as applicable).

1) Pre‑Grad FX IR (ATen FX graph)
   - Start: FX graph from Dynamo.
   - Passes: `torch/_inductor/fx_passes/pre_grad.py:pre_grad_passes()` (predispatch normalization/fusions: split/cat, conv+bn eval, pattern rewrites, etc.).
   - Where invoked: `compile_fx.run_pre_grad_passes()` inside `_compile_fx_main()`.

2) Joint Graph (FX) via AOTAutograd
   - AOT constructs a joint fwd+bwd graph with decompositions (see `select_decomp_table()`), then applies joint passes: `joint_graph_passes()`.
   - Partition fn: default `functorch.compile.min_cut_rematerialization_partition`.
   - Files: `torch/_inductor/compile_fx.py` (setup), `torch/_inductor/fx_passes/joint_graph.py` (passes).

3) Post‑Grad FX IR (partitioned fwd/bwd graphs)
   - Passes: `torch/_inductor/fx_passes/post_grad.py:post_grad_passes()` (DCE, reorder for locality, group‑batch fusions, 2‑stage pattern passes, optional distributed bucketing, etc.).
   - Where invoked: `compile_fx_forward()` / `compile_fx_backward()` → `_recursive_post_grad_passes()`.

4) Inductor IR (lowering from FX)
   - Builder: `GraphLowering` interprets FX and creates Inductor IR nodes (buffers, pointwise, reductions, extern/template nodes, etc.).
     - Files: `torch/_inductor/graph.py` (driver), `torch/_inductor/lowering.py` (per‑op lowers), `torch/_inductor/ir.py` (IR nodes/layouts), `torch/_inductor/sizevars.py` (symbolic shapes).
   - Device/runtime helpers: `torch/_inductor/utils.py`, `torch/_inductor/runtime/*`.

5) Scheduler IR and Fusion
   - Scheduler takes Inductor IR ops and fuses/partitions them into scheduled nodes (fused groups, templates, externs) with dependencies.
   - Files: `torch/_inductor/scheduler.py` (fusion legality, grouping, dependency analysis), `torch/_inductor/dependencies.py`.

6) Codegen
   - Backend registration: `torch/_inductor/codegen/common.py:init_backend_registration()`.
   - On CUDA, `CUDACombinedScheduling` delegates per node to:
     - Triton: `torch/_inductor/codegen/triton.py` (kernels) + `torch/_inductor/runtime/triton_heuristics.py` (autotune, launch).
     - CUDA templates (CUTLASS/C++): `torch/_inductor/codegen/cuda/*.py` (`cuda_kernel.py`, `cuda_template.py`, `cutlass_utils.py`, `cutlass_cache.py`).
     - CuteDSL (optional), ROCm variants.
   - Wrapper (host) code: Python wrapper default `torch/_inductor/codegen/wrapper.py`, or C++ wrapper on GPU `torch/_inductor/codegen/cpp_wrapper_gpu.py` when `config.cpp_wrapper` is enabled.
   - Caching/compilation: `torch/_inductor/codecache.py` (module caching, source emission, logs), `torch/_inductor/async_compile.py` (background compiles).


## CUDA Backends: Triton and CUTLASS

- Triton path (default):
  - Most elementwise/reduction kernels lowered to Triton (`codegen/triton.py`), autotuned via `runtime/triton_heuristics.py` and benchmarkers.
  - GEMMs may also be Triton (including TMA on supported arch) depending on config/heuristics.

- CUTLASS/CUDA template path:
  - Large GEMMs/convs can route to CUDA C++ templates (`codegen/cuda/cuda_kernel.py`, `cuda_template.py`) with CUTLASS selected via `select_algorithm.py` and `utils.use_cutlass_template()`.
  - Enable via config: `torch._inductor.config.max_autotune = True` and include `"CUTLASS"` in `torch._inductor.config.max_autotune_gemm_backends`.
  - CUTLASS discovery/helpers: `codegen/cuda/cutlass_utils.py`, cache `codegen/cuda/cutlass_cache.py`.


## Observability and Introspection

Beyond `TORCH_LOGS` and `torch._inductor.config.trace.enabled`, the following are useful:

- Artifact logs (enable with `torch._logging.set_logs(...)` or `TORCH_LOGS=`):
  - `pre_grad_graphs`, `post_grad_graphs` from `compile_fx.py`
  - `kernel_code` from `async_compile.py` and CUDA/Triton codegens
  - `fusion`, `schedule`, `compute_dependencies` from scheduler/codegen
  - `benchmarking`, `inductor_metrics` from runtime

- Structured artifacts emitted by Inductor:
  - `before_pre_grad_graph`, `after_pre_grad_graph` in `compile_fx.run_pre_grad_passes()`
  - `before_post_grad_graph` and `after_post_grad_graph` around `_recursive_post_grad_passes()`
  - `fx_graph_runnable` recorded via `torch._dynamo.repro.after_aot.save_graph_repro()`

- Debug directory dumps: set `TORCH_COMPILE_DEBUG=1` to turn on `config.trace.enabled` and get a temp debug dir with:
  - `fx_graph_runnable.py`, `fx_graph_readable.py`, `fx_graph_transformed.py`
  - `ir_pre_fusion.txt`, `ir_post_fusion.txt`, `graph_diagram.svg` (post‑fusion), `orig_fx_graph_diagram.svg`
  - Generated sources (wrapper and kernels) when `config.trace.output_code=True`

- Provenance mapping and kernel stack traces: enable `config.trace.provenance_tracking_level` (≥1) to correlate generated kernels to FX nodes. See `torch/_inductor/debug.py`.

- Save/Replay a compile at the inner boundary:
  - Inside `_compile_fx_inner()` Inductor calls `debug.save_args_for_compile_fx_inner(...)`, which you can turn on via logger level. This writes `/tmp/inductor_saved_args/compile_fx_inner_*.pkl`.
  - Replay: `from torch._inductor.debug import load_args_and_run_compile_fx_inner; load_args_and_run_compile_fx_inner(path)`.

- Programmatic peeking:
  - After `GraphLowering.run()`, inspect `graph.scheduler.nodes` (scheduled/fused nodes), and use `torch._inductor.debug.draw_buffers(nodes, fname=...)`.
  - During codegen, backends go through `V.graph.wrapper_code` and `V.kernel`; you can instrument those objects in a local run for deeper insight.


## End‑to‑End Trace (step‑by‑step, internal APIs)

The snippet below decomposes `torch.compile(..., backend="inductor")` into the same stages Inductor runs internally, while letting you hook each stage. It works for CUDA with either Triton or CUTLASS (toggle via configs).

```python
import torch
import torch.nn as nn

from torch._dynamo import optimize
from torch._inductor import config as ind_config
from torch._inductor.compile_fx import (
    run_pre_grad_passes,
    compile_fx_forward,
    compile_fx_backward,
)
from torch._inductor.compile_fx import create_compiler_config_extra
from torch._inductor.decomposition import select_decomp_table
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import min_cut_rematerialization_partition


def inductor_backend_step_by_step(gm, example_inputs):
    # 0) Optional debug config (CUDA example):
    ind_config.trace.enabled = True
    ind_config.trace.output_code = True
    ind_config.pick_loop_orders = True
    # For CUTLASS (optional toggles):
    # ind_config.max_autotune = True
    # ind_config.max_autotune_gemm_backends = "CUTLASS,TRITON,CUBLAS"

    # 1) Pre‑grad FX passes
    gm = run_pre_grad_passes(gm, example_inputs)

    # 2) Prepare decompositions + compiler config holders
    decompositions = select_decomp_table()
    num_example_inputs = len(example_inputs)
    compiler_config_extra = create_compiler_config_extra(ind_config)

    # 3) Define Inductor compilers (as AOTAutograd expects)
    def fw_compiler_base(gm_fw, ex_fw, is_inference):
        from torch._inductor.compile_fx import get_num_model_outputs
        num_orig_model_outputs = get_num_model_outputs(gm)
        return compile_fx_forward(
            gm_fw,
            ex_fw,
            num_orig_model_outputs=num_orig_model_outputs,
            num_example_inputs=num_example_inputs,
            compiler_config_extra=compiler_config_extra,
            is_inference=is_inference,
        )

    # Serialize wrappers so AOT can cache and re‑invoke them safely.
    from torch._functorch.aot_autograd import SerializableAOTDispatchCompiler
    from torch._inductor.output_code import OutputCode
    fw_compiler = SerializableAOTDispatchCompiler(OutputCode, lambda gm_fw, ex_fw: fw_compiler_base(gm_fw, ex_fw, is_inference=False))
    inference_compiler = SerializableAOTDispatchCompiler(OutputCode, lambda gm_fw, ex_fw: fw_compiler_base(gm_fw, ex_fw, is_inference=True))

    def bw_compiler(gm_bw, ex_bw):
        return compile_fx_backward(
            gm_bw,
            ex_bw,
            compiler_config_extra=compiler_config_extra,
        )
    bw_compiler = SerializableAOTDispatchCompiler(OutputCode, bw_compiler)

    # 4) Run AOTAutograd to get compiled fw/bw and assemble a callable
    compiled = aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
        decompositions=decompositions,
        partition_fn=min_cut_rematerialization_partition,
        cudagraphs=compiler_config_extra.cudagraphs,
        boxed_forward_device_index=compiler_config_extra.forward_device,
    )(gm, example_inputs)

    # ‘compiled’ is equivalent to what torch.compile() returns for this frame.
    return compiled


def compiled_model(model: nn.Module):
    # Use Dynamo optimize so we feed the precise FX graph Dynamo would hand to Inductor
    return optimize(inductor_backend_step_by_step)(model)


if __name__ == "__main__":
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(1024, 1024, bias=False)
        def forward(self, x):
            return torch.relu(self.lin(x))

    torch.cuda.init()
    m = compiled_model(M().cuda())
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
    y = m(x)  # runs through all stages; inspect debug dir/logs
```

Notes

- To “peek” extra internals, you can temporarily monkey‑patch `GraphLowering.codegen()` or `Scheduler.codegen()` to capture the IR/schedule before/after fusion.
- Use `torch._logging.set_logs(fusion=True, schedule=True, kernel_code=True, pre_grad_graphs=True, post_grad_graphs=True)` for targeted logs.
- To inspect the joint graph independently, use `from torch._functorch.aot_autograd import aot_export_module` and print the returned `gm` before partitioning.


## Key Files Map

- Entry/Orchestrator: `torch/_inductor/compile_fx.py` (entry `compile_fx`, orchestration `_compile_fx_main`, per‑graph compilers, cuda‑graphs setup)
- Pre‑Grad Passes: `torch/_inductor/fx_passes/pre_grad.py`
- Joint Graph Passes: `torch/_inductor/fx_passes/joint_graph.py`
- Post‑Grad Passes: `torch/_inductor/fx_passes/post_grad.py`
- Lowering/IR: `torch/_inductor/graph.py`, `torch/_inductor/lowering.py`, `torch/_inductor/ir.py`
- Scheduler/Fusion: `torch/_inductor/scheduler.py`, `torch/_inductor/dependencies.py`
- Backends (CUDA):
  - Triton: `torch/_inductor/codegen/triton.py`, `torch/_inductor/runtime/triton_heuristics.py`
  - CUDA templates/CUTLASS: `torch/_inductor/codegen/cuda/*.py` (notably `cuda_kernel.py`, `cuda_template.py`, `cutlass_utils.py`)
  - Combined scheduling: `torch/_inductor/codegen/cuda_combined_scheduling.py`
- Wrappers: `torch/_inductor/codegen/wrapper.py` (Python), `torch/_inductor/codegen/cpp_wrapper_gpu.py` (C++ GPU)
- Caching/Async: `torch/_inductor/codecache.py`, `torch/_inductor/async_compile.py`
- Config: `torch/_inductor/config.py`
- Debugging: `torch/_inductor/debug.py`


## Gotchas / Invariants

- Pre‑grad FX is not fully normalized/functional; prefer post‑grad or joint‑graph passes for complex transforms.
- Shape logic relies on symbolic `ShapeEnv`; lowering assumes valid fake‑tensor propagation.
- On CUDA, kernels may be a mix of Triton and CUDA templates; scheduler chooses per‑node backend.
- CUTLASS requires availability of the CUTLASS toolkit and enabling via config; otherwise it is skipped.


## Appendix: Useful Flags

- `TORCH_COMPILE_DEBUG=1` – master switch that turns on `config.trace.enabled`, writes debug artifacts.
- `TORCH_LOGS=+fusion,+schedule,+kernel_code,+pre_grad_graphs,+post_grad_graphs` – enable artifact logs.
- `torch._inductor.config.max_autotune = True` – allow autotuning (try Triton/CUBLAS/CUTLASS per op).
- `torch._inductor.config.max_autotune_gemm_backends = "CUTLASS,TRITON,CUBLAS"` – include CUTLASS in GEMM search.
- `torch._inductor.config.triton.autotune_at_compile_time = True` – autotune during compile.

