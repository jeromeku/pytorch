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
from textwrap import dedent

# ======================================================================================
# 1) CUDA extension
#    - Registers a dispatcher op: my_ns::scale (CUDA + Meta)
#    - Exposes a RAW pybind launcher: scale_cuda_raw_(x, y, a)  <-- used by Inductor
#      The raw launcher writes into 'y' and DOES NOT allocate (so Inductor owns buffers).
# ======================================================================================


ext = load(
    name="my_scale_ext_extern",
    sources=["scale.cu"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)

scale = torch.ops.my_ns.scale            # dispatcher-backed op (eager)
scale_raw = ext.scale_cuda_raw_          # raw launcher (no dispatcher)

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
# from torch._inductor.lowering import register_lowering
# from torch._inductor import ir
# from torch._inductor.ir import ExternKernel

# @register_lowering(scale.default)
# def lowering_my_scale_to_extern(x, a):
#     # Allocate output once (owned by Inductor)
#     y = ir.CreateTensorLike(x, "y")
#     n = ir.ShapeBox(x.get_size().numel())  # not strictly needed for launcher, but useful for guards

#     # Define the runtime call the compiled graph will execute
#     def kernel_fn(X_tensor, Y_tensor, A_scalar):
#         # This is executed in the compiled graph runtime on Python side,
#         # and directly calls our raw CUDA launcher (no dispatcher).
#         scale_raw(X_tensor, Y_tensor, float(A_scalar.item()) if torch.is_tensor(A_scalar) else float(A_scalar))

#     # Wire it up as an ExternKernel
#     return ExternKernel(
#         ["X", "A"], ["Y"], kernel_fn,
#         args=[x, a], out=[y],
#         device="cuda",
#         grid=lambda META: (1,),  # not used by our launcher, but required by interface
#     ).outputs[0]

# # ======================================================================================
# # 3) Two modules:
# #    A) allow_in_graph wrapper → re-enters DISPATCHER at runtime
# #    B) direct custom-op call → lowered to EXTERN → dispatcher-free hot path
# # ======================================================================================

def python_wrapper_allow(x, a: float):
    return scale(x, a)  # calls dispatcher
python_wrapper_allow = torch.compiler.allow_in_graph(python_wrapper_allow)

class ModAllow(torch.nn.Module):
    def __init__(self, a: float): super().__init__(); self.a = a
    def forward(self, x): return python_wrapper_allow(x, self.a)

# class ModLowered(torch.nn.Module):
#     def __init__(self, a: float): super().__init__(); self.a = a
#     def forward(self, x): return scale(x, self.a)

# # ======================================================================================
# # 4) Diagnostics: dispatch table, microbench, and profiler hints
# # ======================================================================================

def dump_dispatch(opname="my_ns::scale"):
    print("=" * 80)
    print(f"DISPATCH TABLE for {opname}")
    print("-" * 80)
    print(torch._C._dispatch_dump_table(opname))

def test_mod(mod, inp, label):
    mod = torch.compile(mod, fullgraph=True, dynamic=False)
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
    dump_dispatch("my_ns::scale")

    N = 4_194_304  # 4M elems; tweak smaller to accentuate CPU overhead
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    alpha = 1.234
    allow_in_graph_mod = ModAllow(alpha)
    test_mod(allow_in_graph_mod, x, "allow_in_graph")

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
