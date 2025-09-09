import torch
from torch._inductor.utils import run_and_get_triton_code

def pointwise(x): return (x * x).relu()

def gemm(a, b): return (a @ b).relu()

def main():
    X = torch.randn(1<<20)
    with torch._inductor.config.patch({"triton.autotune_at_compile_time": False}):
        c0 = run_and_get_triton_code(torch.compile(pointwise), X)
    with torch._inductor.config.patch({"triton.autotune_at_compile_time": True}):
        c1 = run_and_get_triton_code(torch.compile(pointwise), X)
    print("len(defer)", len(c0), "len(ct)", len(c1))

    a = torch.randn(128, 64); b = torch.randn(64, 128)
    with torch._inductor.config.patch({"triton.use_block_ptr": False}):
        g0 = run_and_get_triton_code(torch.compile(gemm), a, b)
    with torch._inductor.config.patch({"triton.use_block_ptr": True}):
        g1 = run_and_get_triton_code(torch.compile(gemm), a, b)
    print("tl.int64 present?", "tl.int64" in g0, "→", "tl.int64" in g1)

if __name__ == "__main__":
    main()

