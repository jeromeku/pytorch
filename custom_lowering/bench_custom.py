import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Turn these on when you want the guts:
# os.environ["TORCH_LOGS"] = "dynamo,inductor,graph,guards"
# os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

import math
import time
import torch
import torch._dynamo as dynamo
from torch.utils.cpp_extension import load
from torch._logging import set_logs
import logging
import torch._inductor.config as config

config.force_disable_caches = True

set_logs(aot_graphs=True)

ext = load(
    name="my_scale_ext_extern",
    sources=["scale.cu"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)

scale = torch.ops.my_ns.scale  # dispatcher-backed op (eager)
scale_raw = ext.scale_cuda_raw_  # raw launcher (no dispatcher)

# Quick sanity check eager
x_chk = torch.randn(8, device="cuda", dtype=torch.float32)
y_chk = torch.empty_like(x_chk)
scale_raw(x_chk, y_chk, 2.0)
assert torch.allclose(y_chk, x_chk * 2.0)
assert torch.allclose(scale(x_chk, 2.0), x_chk * 2.0)

# # ======================================================================================
# # 2) TorchInductor lowering (EXTERN CALL)
# #    We lower my_ns::scale to an ExternKernel that calls scale_cuda_raw_ directly.
# #    The compiled region invokes our raw launcher; the dispatcher is NOT touched.
# # ======================================================================================
from torch._inductor.lowering import register_lowering
from torch._inductor import ir
from torch._inductor.ir import ExternKernel

@register_lowering(scale.default)
def lowering_my_scale_to_extern(x, a):
    # Allocate output once (owned by Inductor)
    y = ir.CreateTensorLike(x, "y")
    n = ir.ShapeBox(x.get_size().numel())  # not strictly needed for launcher, but useful for guards

    # Define the runtime call the compiled graph will execute
    def kernel_fn(X_tensor, Y_tensor, A_scalar):
        # This is executed in the compiled graph runtime on Python side,
        # and directly calls our raw CUDA launcher (no dispatcher).
        scale_raw(X_tensor, Y_tensor, float(A_scalar.item()) if torch.is_tensor(A_scalar) else float(A_scalar))

    # Wire it up as an ExternKernel
    return ExternKernel(
        ["X", "A"], ["Y"], kernel_fn,
        args=[x, a], out=[y],
        device="cuda",
        grid=lambda META: (1,),  # not used by our launcher, but required by interface
    ).outputs[0]

def wrapper_allow_custom(x, a: float):
    return scale(x, a)  # calls dispatcher


def wrapper_allow_raw(x, a: float):
    return scale_raw(x, a)


wrapped_custom = torch.compiler.allow_in_graph(wrapper_allow_custom)
wrapped_raw = torch.compiler.allow_in_graph(wrapper_allow_raw)

class ScaleMod(torch.nn.Module):
    def __init__(self, a: float, impl: str = "custom"):
        super().__init__()
        self.a = a
        self.impl = impl

    def forward(self, x):
        impl = self.impl
        if impl == "custom":
            return scale(x, self.a)
        elif impl == "wrapped_custom":
            return wrapped_custom(x, self.a)
        elif impl == "custom_default":
            return scale.default(x, self.a)

def dump_dispatch(opname="my_ns::scale"):
    print("=" * 80)
    print(f"DISPATCH TABLE for {opname}")
    print("-" * 80)
    print(torch._C._dispatch_dump_table(opname))


def test_mod(mod, inp, label):
    mod = torch.compile(mod, fullgraph=True, dynamic=False)

    print(f"Running {label}...")
    mod(inp)


# def bench(mod, inp, label, iters=300, warmup=80):
#     mod = torch.compile(mod, fullgraph=True, dynamic=False)
#     for _ in range(warmup):
#         mod(inp)
#     torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     for _ in range(iters):
#         mod(inp)
#     torch.cuda.synchronize()
#     dt = time.perf_counter() - t0
#     ips = iters / dt
#     print(f"{label:>28}: {ips:.1f} it/s   (iters={iters})")


def main():
    # dump_dispatch("my_ns::scale")

    N = 4_194_304  # 4M elems; tweak smaller to accentuate CPU overhead
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    alpha = 1.234

    for impl in ["custom_default"]:
        mod = ScaleMod(alpha, impl=impl)    

        test_mod(mod, x, label=impl)
        print(" ===================================================================================== ")
    # print("\nCompiling with fullgraph=True (dispatcher vs extern) ...\n")
    # bench(ModAllow(1.234), x, "allow_in_graph → dispatcher")
    # bench(ModLowered(1.234), x, "custom_op lowering → EXTERN")

    # print("\nTips to *see* the path differences:\n"
    #       "  1) Logs:\n"
    #       "     TORCH_LOGS=dynamo,inductor,graph,guards python build_and_run_extern.py\n"
    #       "  2) C++ stacks when dispatcher is hit:\n"
    #       "     TORCH_SHOW_CPP_STACKTRACES=1 python build_and_run_extern.py\n"
    #       "  3) Profiler (CPU+CUDA) around a short loop: the allow_in_graph path\n"
    #       "     will show extra CPU frames entering the dispatcher; the extern path\n"
    #       "     should be lean (mostly launch from compiled region).\n")


if __name__ == "__main__":
    main()
