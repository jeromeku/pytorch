TorchInductor Pipeline (default torch.compile backend)

Scope: FX passes → Lowering to Inductor IR → Scheduling/Fusion → Codegen

High-level order (compile_fx)
- Entry: `compile_fx(model_, example_inputs_, ...)` orchestrates AOTAutograd and Inductor. Ref: torch/_inductor/compile_fx.py:2367–2681.
- Pre-grad FX passes (on the forward graph before AOTAutograd): `_recursive_pre_grad_passes` → `pre_grad_passes`. Ref: 490–506 (recursive), 268–360 in pre_grad.py.
- AOTAutograd generates joint graph / fwd/bwd partitions. Inductor then runs joint-graph FX passes: `_recursive_joint_graph_passes` → `joint_graph_passes`. Ref: 509–527 (recursive), joint_graph.py:561–619.
- Post-grad FX passes run per fwd/bwd graph: `_recursive_post_grad_passes` → `post_grad_passes`. Ref: 529–538 (recursive), post_grad.py:83–140 and beyond.
- Lowering to Inductor IR: `GraphLowering(...).run(*example_inputs)` converts FX nodes via registered lowerings. Ref: torch/_inductor/compile_fx.py:1389–1449; torch/_inductor/graph.py:935–957, 1180–1291; torch/_inductor/lowering.py (registry at ~469–488 and many lowerings).
- Scheduling/Fusion: `Scheduler(self.operations)` builds a scheduler graph, computes deps, fuses nodes, and reorders loops. Ref: torch/_inductor/graph.py:2254; torch/_inductor/scheduler.py:2065–2149, 2135–2149, 2769–2800.
- Codegen: `graph.codegen()` or `codegen_with_cpp_wrapper()` emits Python/CUDA/Triton/C++ wrappers, possibly autotuning. Ref: torch/_inductor/compile_fx.py:1444–1485, 1463–1471; torch/_inductor/graph.py:2131–2170, 2256–2274.

IRs at each phase
- FX Graph (ATen/Prims) — input to Inductor passes.
- Inductor IR (tensor/compute buffers, ops) — produced by `GraphLowering.call_function` via `lowerings[...]`. Ref: torch/_inductor/graph.py:1180–1291; torch/_inductor/lowering.py.
- Scheduler graph (SchedulerNodes/FusedSchedulerNodes) — fusion and ordering over IR ops. Ref: torch/_inductor/scheduler.py:2065–2149, 1330–1712.
- Output code (Python invoking Triton/CUDA kernels or C++ wrappers) — final artifacts.

FX pass details (files + anchors)
- pre_grad_passes: torch/_inductor/fx_passes/pre_grad.py:268–360
  - Normalization, split/cat rewrites, numpy compatibility, conv+bn fusion (eval), group-batch fusions, and optional custom/user passes.
  - Example formation of PatternMatcherPasses: lines 37–66; application order: 300–331.
- joint_graph_passes: torch/_inductor/fx_passes/joint_graph.py:561–619
  - Canonicalize aten IR first (553–560), remove no-ops, optional constant folding, pattern passes in stages (597–603), RNG replacement if not falling back.
- post_grad_passes: torch/_inductor/fx_passes/post_grad.py:83–140, 270–294
  - DCE (98–101), locality reordering (102–105), custom pre/post hooks (109–112, 609–613), communication bucketing (200–236), and final mutation-introducing reinplacement and HOP decomps kept last (270–290).

Lowering and scheduling
- Lowering dispatch: FX `call_function` mapped to lowering impls via `lowerings` dict, with optional layout constraints and fallbacks. Ref: torch/_inductor/graph.py:1180–1291; torch/_inductor/lowering.py:469–488, 1601–1610, 1945–2009.
- Scheduler init → deps/toposort → fusion → loop reordering → memory planning hooks. Ref: torch/_inductor/scheduler.py:2071–2149, 2135–2149, 2728–2767, 2773–2800, 2562–2600.
- Debug IR snapshots: `log_ir_pre_fusion` / `log_ir_post_fusion`. Ref: torch/_inductor/debug.py:709–721.

Experiment quickly
- Script: scripts/run_fx_pass_trace.py — run pre/joint/post passes individually on a toy graph and diff the printed FX.
- Script: scripts/run_inductor_ir_trace.py — lower to Inductor IR, print IR before and after fusion, and emit code.

