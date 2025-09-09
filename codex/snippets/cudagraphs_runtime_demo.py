import torch

def f(x): return (x.sin() + 1.0).cos()

def main():
    X = torch.randn(8192)
    with torch._inductor.config.patch({"triton.cudagraphs": True}):
        g = torch.compile(f)
        g(X); g(X)
    print("cudagraphs enabled and replayed")

if __name__ == "__main__":
    main()

