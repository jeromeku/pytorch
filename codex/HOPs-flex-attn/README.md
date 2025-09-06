Overview

- Goal: Explain how Higher-Order Ops (HOPs) work in PyTorch and trace, end-to-end, how FlexAttention is implemented and compiled to an Inductor kernel via torch.compile. This folder collects a deep call-path, annotated code references, and a runnable minimal example.
- Repo snapshot date: 2025-09-06 (local working copy). Paths and line numbers below refer to this checkout.

Contents

- call_path.md: Step-by-step call path from user API → HOPs → Dynamo/FX capture → Inductor lowering → Triton/CPU codegen, with file:line references and snippets.
- minimal_example.py: Tiny, runnable FlexAttention example showing how HOPs are invoked through torch.compile, with hints to surface debug IR.
- trace_notes.md: How to run and what debug artifacts to inspect (FX graphs, Inductor IR, kernel sources).

Quick Start

- Run: `python codex/HOPs-flex-attn/minimal_example.py`
- Tip: For maximal logs and debug dumps, set env:
  - `TORCH_COMPILE_DEBUG=1`
  - `TORCH_LOGS=+dynamo,inductor,aot`
  - Optional: `TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1` and/or `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`

