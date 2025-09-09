import torch
from torch.testing._internal.logging_utils import logs_to_string

def split_cat(x):
    a, b = torch.tensor_split(x, 2, dim=-1)
    return torch.cat([a, b], dim=-1)

def main():
    ls, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
    with ctx():
        torch.compile(split_cat)(torch.randn(4, 8))
    print(ls.getvalue())

if __name__ == "__main__":
    main()

