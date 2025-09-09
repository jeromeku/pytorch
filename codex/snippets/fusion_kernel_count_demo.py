import torch
from torch._inductor.utils import run_and_get_kernels

def f(x):
    y = x.sin(); z = y * 2 + y.cos(); return z.relu()

def main():
    X = torch.randn(1 << 16)
    with torch._inductor.config.patch({"max_fusion_size": 64}):
        _, k1 = run_and_get_kernels(torch.compile(f), X)
    with torch._inductor.config.patch({"max_fusion_size": 1}):
        _, k2 = run_and_get_kernels(torch.compile(f), X)
    print("kernels:", len(k1), "→", len(k2))

if __name__ == "__main__":
    main()

