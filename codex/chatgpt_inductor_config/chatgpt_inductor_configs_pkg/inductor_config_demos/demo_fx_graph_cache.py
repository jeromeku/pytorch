#!/usr/bin/env python
import os, time, torch
try:
    from torch._inductor.utils import run_and_get_code, fresh_cache
except Exception:
    run_and_get_code = None
    class fresh_cache:
        def __enter__(self): return None
        def __exit__(self,*a): return False
ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts"); os.makedirs(ART_DIR, exist_ok=True)
def dump_code(tag, fn, *args):
    if run_and_get_code is None:
        fn(*args); return "// run_and_get_code missing"
    with fresh_cache(): code = run_and_get_code(fn, *args)
    torch.compile(fn, backend="inductor", fullgraph=True)(*args)
    open(os.path.join(ART_DIR, f"{tag}.py"),"w").write(code if isinstance(code,str) else str(code))
    return code


from torch._dynamo.testing import CompileCounterWithBackend
def f(x): return x*2
def run(use_cache, tag):
    torch._inductor.config.fx_graph_cache=use_cache
    torch._inductor.config.fx_graph_remote_cache=False
    backend=CompileCounterWithBackend("inductor")
    g=torch.compile(f, backend=backend, fullgraph=True); x=torch.randn(2,3)
    g(x); g(x); g(x)
    open(os.path.join(ART_DIR, f"fx_cache_{tag}.txt"),"w").write(f"frames={backend.frame_count}\n")
    print(tag, "frames", backend.frame_count)
if __name__=="__main__":
    run(False,"off"); run(True,"on")
