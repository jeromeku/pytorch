import torch
from torch._dynamo.utils import counters

def branchy(x, t):
    return x.sin() if t > 0 else x.cos()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, device=device)
    counters.clear()
    g = torch.compile(branchy)
    g(x, 1); g(x, 1)
    print("hits:", counters["inductor"].get("fxgraph_cache_hit", 0))
    g(x, -1)
    print("misses:", counters["inductor"].get("fxgraph_cache_miss", 0))

if __name__ == "__main__":
    main()

