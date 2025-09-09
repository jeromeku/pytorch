import torch
from torch._dynamo.utils import counters

def f(x, s): return x * s

def main():
    x = torch.randn(8)
    counters.clear()
    g = torch.compile(f, fullgraph=True)
    g(x, 2); g(x, 2)
    print("hits:", counters["inductor"]["fxgraph_cache_hit"], "misses:", counters["inductor"]["fxgraph_cache_miss"])
    g(x, 3)
    print("after guard change misses:", counters["inductor"]["fxgraph_cache_miss"])

    g = torch.compile(lambda t: (t + 1).relu())
    g(x)
    artifacts = torch.compiler.save_cache_artifacts()
    torch.compiler.load_cache_artifacts(artifacts[0])
    print("hot-loaded artifacts")

if __name__ == "__main__":
    main()

