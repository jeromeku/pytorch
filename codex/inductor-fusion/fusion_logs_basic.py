import torch
from torch.testing._internal.logging_utils import logs_to_string

def f(x):
    y = x * 2
    z = y + x
    return z.relu()

def main():
    log, ctx = logs_to_string("torch._inductor.scheduler", "fusion")
    with ctx():
        torch.compile(f)(torch.randn(8192))
    print(log.getvalue())

if __name__ == "__main__":
    main()

