import pytest
import importlib
import torch
from torch.fx import symbolic_trace
from tests.conftest import skip_if_no_gpu, skip_if_no_trtllm, count_nodes_by_target, resolve_pass_apply

AUTO_DEPLOY_FUNC_FIX = "tensorrt_llm._torch.auto_deploy.transform.library.functionalization"
LEGACY_FUNC_FIX = "tensorrt_llm._torch.compilation.fix_functionalization"

@skip_if_no_gpu
@skip_if_no_trtllm
def test_fix_functionalization_keeps_fused_ops():
    # Resolve functionalization fix pass
    apply_fn, used = resolve_pass_apply(AUTO_DEPLOY_FUNC_FIX, LEGACY_FUNC_FIX)
    if apply_fn is None:
        pytest.skip(f"No functionalization-fix pass with 'apply' in: {AUTO_DEPLOY_FUNC_FIX}, {LEGACY_FUNC_FIX}")

    # Build a toy graph that wraps a fused op through auto_functionalize
    try:
        auto_func = torch._higher_order_ops.auto_functionalize.auto_functionalized
    except Exception as e:
        pytest.skip(f"PyTorch auto_functionalize not available: {e}")

    def toy(x, r, w):
        at = auto_func(
            torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default,
            input=x, residual=r, weight=w, eps=1e-5,
        )
        return at[0]

    class M(torch.nn.Module):
        def forward(self, x, r, w):
            return toy(x, r, w)

    m = M().cuda().eval()
    gm = symbolic_trace(m)

    before_fused = count_nodes_by_target(gm, "trtllm.flashinfer_fused_add_rmsnorm")
    gm_after = apply_fn(gm)
    after_fused = count_nodes_by_target(gm_after, "trtllm.flashinfer_fused_add_rmsnorm")

    assert after_fused >= before_fused, f"Functionalization fix from '{used}' reduced fused-op visibility"
