# TensorRT-LLM PyTorch backend (main) — pass tests + MultiStream/Events demos

This bundle targets the **current `main` branch** of [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). It contains:

- **Unit tests** for torch.compile passes (norm fusion, functionalization-fix, plus a generic pass harness).
- **Examples** for CUDA **MultiStream** overlap with **Events** (CUDA events for inter-stream dependencies) and a **KV-cache Events** lifecycle demo.
- A tiny **before/after** driver that prints the FX graph around a selected pass.

> **Context from our discussion** (short version):  
> The small passes that massage the FX graph for the PyTorch backend have been moving. Historically they lived under `tensorrt_llm/_torch/compilation/`. On **main**, the bulk of the pattern-based transforms (like RMSNorm fusion) are commonly wired through the **AutoDeploy** optimizer stack under `tensorrt_llm/_torch/auto_deploy/transform/library/`. That’s why these tests search **AutoDeploy first** and the legacy `compilation` directory second. MultiStream scheduling and CUDA events are **runtime concerns** (scheduler/executor), not something these FX passes directly emit; the passes mainly help by producing **bigger, fewer kernels** to make overlap easier.

## What’s covered

### Passes (on `main`)
- **RMSNorm fusion**: collapses `add + rms_norm` into a fused `trtllm.flashinfer_fused_add_rmsnorm` op.  
  AutoDeploy path: `tensorrt_llm/_torch/auto_deploy/transform/library/rms_norm.py` (applies patterns via `torch._inductor.pattern_matcher`).  
  Legacy path (if present): `tensorrt_llm/_torch/compilation/fuse_norm.py` or `norm_fusion.py`.

- **Functionalization fix**: maintains custom/fused op boundaries across PyTorch **functionalization** (so Inductor pattern matching isn’t defeated).  
  AutoDeploy path (if present): `tensorrt_llm/_torch/auto_deploy/transform/library/functionalization.py`.  
  Legacy path: `tensorrt_llm/_torch/compilation/fix_functionalization.py`.

The tests dynamically import the most likely modules and **skip** if the pass is not present for your commit of `main`.

### MultiStream vs. Events
- **MultiStream**: Backend/runtime dispatches work to multiple CUDA streams; **passes do not schedule** streams—they just simplify graphs for better fusion and fewer sync points.
- **CUDA Events**: Used for inter-stream dependencies (see `examples/multistream_overlap.py`).  
- **KV-cache “Events”** in TRT-LLM are **lifecycle callbacks** (Created/Updated/Removed/Stored) for KV-cache management—**not** CUDA events. See `examples/kv_cache_events_demo.py` for a minimal illustration using the real manager if available or a shim.

## References (canonical links)

- **PyTorch backend overview**: https://nvidia.github.io/TensorRT-LLM/torch.html
- **Adding a model; `RMSNorm`**: https://nvidia.github.io/TensorRT-LLM/torch/adding_new_model.html
- **Release notes** (ongoing refactors toward modular optimizer / transforms): https://nvidia.github.io/TensorRT-LLM/release-notes.html
- **AutoDeploy transform (rms_norm) stack trace on `main`** showing `patterns.apply(graph)` and file path:  
  https://github.com/NVIDIA/TensorRT-LLM/issues/7270#issue-2585418081

> For **line references** that match your exact commit of `main`, use ripgrep locally:  
> ```bash
> rg -n "class RMSNorm|flashinfer_fused_add_rmsnorm|pattern_matcher|apply\(" -S tensorrt_llm/_torch
> ```
> Then copy the `blob/<commit>/<path>#L<start>-L<end>` URLs from the GitHub UI.

## Layout

```
.
├── README.md
├── examples
│   ├── kv_cache_events_demo.py
│   ├── multistream_overlap.py
│   └── run_pass_before_after.py
└── tests
    ├── conftest.py
    ├── test_pass_fix_functionalization.py
    ├── test_pass_fuse_rmsnorm.py
    └── test_passes_generic.py
```

## Quick start

```bash
# In your TRT-LLM development venv (with CUDA & GPU)
pip install -U pytest rich

# Run tests
pytest -q tests

# Show before/after for a specific pass (tries AutoDeploy first, then legacy)
python examples/run_pass_before_after.py --pass fuse_norm
python examples/run_pass_before_after.py --pass fix_functionalization

# See CUDA MultiStream overlap (use nsys for a timeline)
python examples/multistream_overlap.py

# Observe KV-cache Events lifecycle (real manager if importable; shim otherwise)
python examples/kv_cache_events_demo.py
```

## Notes

- Operator names and module locations can **change on `main`**. These tests intentionally check multiple import paths and mark **xfail** if a recognizer doesn’t match the exact pattern in your build.
- The fused op substring this pack looks for is `trtllm.flashinfer_fused_add_rmsnorm` (or just `flashinfer_fused_add_rmsnorm`). If your commit uses a different symbol, tweak `test_pass_fuse_rmsnorm.py`.
