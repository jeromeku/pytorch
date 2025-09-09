import torch
from torch.testing._internal.logging_utils import logs_to_string

def g(x):
    y = x.roll(1, dims=0)
    z = y + x
    return z

def main():
    log, ctx = logs_to_string("torch._inductor.scheduler", "fusion")
    with ctx():
        torch.compile(g)(torch.randn(8192))
    print(log.getvalue())

if __name__ == "__main__":
    main()

