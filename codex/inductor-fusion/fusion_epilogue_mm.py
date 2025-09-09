import torch
from torch._inductor.utils import run_and_get_kernels

def mm_relu(a, b):
    return (a @ b).relu()

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(1024, 1024, device=dev)
    b = torch.randn(1024, 1024, device=dev)
    _, kernels = run_and_get_kernels(torch.compile(mm_relu), a, b)
    print("num kernels:", len(kernels))

if __name__ == "__main__":
    main()

