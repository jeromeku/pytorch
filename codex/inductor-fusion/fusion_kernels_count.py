import torch
from torch._inductor.utils import run_and_get_kernels

def h(x):
    a = x.sin(); b = a * 2; c = b + a; return c.relu()

def main():
    X = torch.randn(1 << 18)
    with torch._inductor.config.patch({"max_fusion_size": 64}):
        _, k1 = run_and_get_kernels(torch.compile(h), X)
    with torch._inductor.config.patch({"max_fusion_size": 1}):
        _, k2 = run_and_get_kernels(torch.compile(h), X)
    print("kernels fused:", len(k1), " vs limited:", len(k2))

if __name__ == "__main__":
    main()

