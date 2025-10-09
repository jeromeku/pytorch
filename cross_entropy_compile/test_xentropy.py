import torch
import torch.nn.functional as F
from torch.testing._internal.logging_utils import multiple_logs_to_string
from contextlib import contextmanager, redirect_stderr, redirect_stdout, ExitStack
from quack.cross_entropy import cross_entropy_fwd, cross_entropy

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

import logging
from torch._inductor import config

# torch._logging.set_logs(inductor=logging.DEBUG)

config.enable_auto_functionalized_v2 = True
config.trace.enabled = False
config.force_disable_caches = True

M = 2
N = 32768
input_dtype = torch.bfloat16
use_compile = True
inplace_backward = False
experimental_bwd = True


@contextmanager
def inductor_pre_post_graph_ctx(label: str):
    (pregrad, postgrad), ctx = multiple_logs_to_string(
        "torch._inductor.compile_fx", "pre_grad_graphs", "post_grad_graphs"
    )
    with ctx():
        yield

    print(f"{label}")
    print(f"{pregrad.getvalue().strip()}")
    print(f"{postgrad.getvalue().strip()}")


def test_cross_entropy(M, N, input_dtype, inplace_backward, use_compile, experimental_bwd, fullgraph: bool = True):
    """Test Cross Entropy forward pass against reference implementation."""
    device = "cuda"
    atol, rtol = 5e-5, 1e-5
    torch.random.manual_seed(0)

    # Create input tensors (scale down to avoid overflow)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)

    # Forward pass
    function = torch.compile(cross_entropy, fullgraph=fullgraph) if use_compile else cross_entropy

#    with inductor_pre_post_graph_ctx("XENTROPY_FWD"):
    loss = function(
        x,
        target,
        reduction="none",
        inplace_backward=inplace_backward,
        # experimental_bwd=experimental_bwd,
    )

    dloss = torch.randn_like(loss)
    torch.cuda.synchronize()
    # breakpoint()

    with inductor_pre_post_graph_ctx("XENTROPY_BWD"):
        (dx,) = torch.autograd.grad(loss, x, grad_outputs=dloss)
    # breakpoint()

    x_ref = x.detach().clone().requires_grad_()
    target_ref = target.detach().clone()

    # loss_ref = F.cross_entropy(x_ref.float(), target_ref, reduction="none")
    # # Check output shape and dtype
    # assert loss.shape == (M,)
    # assert loss.dtype == torch.float32
    # # Check accuracy
    # torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
    # # Check cross entropy properties
    # # All values should be non-negative
    # assert (loss >= 0).all()
    # # Check that loss is reasonable (not inf or nan)
    # assert not torch.isnan(loss).any()
    # assert not torch.isinf(loss).any()
    # # Test backward pass
    # torch.cuda.synchronize()

    # (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    # (dx,) = torch.autograd.grad(loss, x, grad_outputs=dloss)
    # assert dx.shape == x.shape
    # torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)


@contextmanager
def stderr_stdout_redirect_ctx(err_file, out_file):
    with open(err_file, "w") as _err, open(out_file, "w") as _out:
        with redirect_stderr(_err), redirect_stdout(_out):
            yield


prefix = ""
if use_compile:
    prefix += "compile"
if inplace_backward:
    prefix += "inplace"
if experimental_bwd:
    prefix += "experimental"
import cutlass
cutlass.cuda.initialize_cuda_context()
#device()
err_log, out_log = [f"{prefix}.{t}.log" for t in ("err", "out")]
#with stderr_stdout_redirect_ctx(err_log, out_log):
test_cross_entropy(
    M,
    N,
    input_dtype,
    inplace_backward=True,
    use_compile=False,
    experimental_bwd=False,
)

# test_cross_entropy_fwd_with_grad()

# def test_cross_entropy_fwd_with_grad(
#     M=2, N=32768, input_dtype=torch.bfloat16, inplace_backward=True, use_compile=True
# ):
#     """Test Cross Entropy forward pass with gradient computation."""
#     device = "cuda"
#     atol, rtol = 1e-4, 1e-4
#     torch.random.manual_seed(0)
#     x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
#     target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
#     x_ref = x.detach().clone().requires_grad_()
#     target_ref = target.detach().clone()

#     # Test forward with gradient computation
#     function = (
#         torch.compile(cross_entropy_fwd, fullgraph=True) if use_compile else cross_entropy_fwd
#     )

#     (pregrad, postgrad), ctx = multiple_logs_to_string(
#         "torch._inductor.compile_fx", "pre_grad_graphs", "post_grad_graphs"
#     )

#     with ctx():
#         if inplace_backward:
#             x_copy = x.detach().clone()
#             loss, lse, dx = function(
#                 x_copy, target, return_lse=True, return_dx=True, inplace_backward=True
#             )
#             # Check that dx is the same tensor as x_copy (inplace)
#             assert dx is x_copy, "inplace_backward should modify x in-place"
#         else:
#             loss, lse, dx = function(
#                 x, target, return_lse=True, return_dx=True, inplace_backward=False
#             )
#             # Check that dx is a different tensor from x
#             assert dx is not x, "non-inplace should create new tensor"

#     print(f"{pregrad.getvalue().strip()}")
#     print(f"{postgrad.getvalue().strip()}")

#     breakpoint()
#     # Reference implementation
#     loss_ref = F.cross_entropy(x_ref.float(), target_ref, reduction="none")
#     lse_ref = torch.logsumexp(x_ref.float(), dim=-1)
#     dloss = torch.ones_like(loss_ref)  # Need dloss to be 1.0
#     (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)

#     # # Check results
#     # torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
#     # torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
#     # torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)
