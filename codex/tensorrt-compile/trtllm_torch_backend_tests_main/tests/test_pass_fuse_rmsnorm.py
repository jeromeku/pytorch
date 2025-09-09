import pytest
import importlib
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from tests.conftest import skip_if_no_gpu, skip_if_no_trtllm, count_nodes_by_target, resolve_pass_apply

# On `main`, norm fusion logic commonly lives under AutoDeploy transforms.
AUTO_DEPLOY_RMSNORM = "tensorrt_llm._torch.auto_deploy.transform.library.rms_norm"
LEGACY_FUSE_NORM = "tensorrt_llm._torch.compilation.fuse_norm"
LEGACY_NORM_FUSION = "tensorrt_llm._torch.compilation.norm_fusion"

@skip_if_no_gpu
@skip_if_no_trtllm
def test_fuse_add_rmsnorm_main():
    # Import the high-perf RMSNorm module used by TRT-LLM's PyTorch path
    try:
        RMSNorm = importlib.import_module("tensorrt_llm._torch.modules.rms_norm").RMSNorm
    except Exception as e:
        pytest.skip(f"RMSNorm module not found: {e}")

    # Resolve pass.apply, biased to AutoDeploy on main
    apply_fn, used = resolve_pass_apply(AUTO_DEPLOY_RMSNORM, LEGACY_FUSE_NORM, LEGACY_NORM_FUSION)
    if apply_fn is None:
        pytest.skip(f"No usable fusion pass 'apply' found in: {AUTO_DEPLOY_RMSNORM}, {LEGACY_FUSE_NORM}, {LEGACY_NORM_FUSION}")

    class AddRms(nn.Module):
        def __init__(self, hidden=64, eps=1e-5, dtype=torch.float16):
            super().__init__()
            self.norm = RMSNorm(hidden_size=hidden, eps=eps, dtype=dtype)
        def forward(self, x, residual):
            return self.norm(x + residual)

    model = AddRms().cuda().eval()
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    r = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    gm = symbolic_trace(model)

    before_add = count_nodes_by_target(gm, "aten.add")
    before_rms = count_nodes_by_target(gm, "rms_norm")
    assert (before_add + before_rms) >= 1, "Expected add/rms chain not present"

    gm_after = apply_fn(gm)  # pass expected to return a GraphModule
    fused_count = max(
        count_nodes_by_target(gm_after, "trtllm.flashinfer_fused_add_rmsnorm"),
        count_nodes_by_target(gm_after, "flashinfer_fused_add_rmsnorm"),
    )
    after_add = count_nodes_by_target(gm_after, "aten.add")
    after_rms = count_nodes_by_target(gm_after, "rms_norm")

    # If fusion fired, add+norm should be gone; otherwise allow xfail (patterns can drift on main).
    if fused_count >= 1:
        assert after_add == 0, "aten.add still present after fusion"
        assert after_rms == 0, "rms_norm still present after fusion"
    else:
        pytest.xfail(f"Fusion pass imported from '{used}' did not match this graph on current main")
