import torch
from torch.testing._internal.logging_utils import logs_to_string

def inplace(x):
    y = x.clone(); y.add_(1); return y

def mm_chain(a, b):
    return (a @ b).relu()

def main():
    print("== functionalization ==")
    print(torch.compile(inplace)(torch.randn(4)))

    print("== post-grad decomposition log ==")
    ls, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
    with ctx():
        torch.compile(mm_chain)(torch.randn(32,64), torch.randn(64,32))
    print(ls.getvalue())

if __name__ == "__main__":
    main()

