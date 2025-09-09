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


def elt(x): return (x.sin()*x.cos()).relu()
def run(cg, tag):
    torch._inductor.config.triton.cudagraphs = cg
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    dev="cuda" if torch.cuda.is_available() else "cpu"
    x=torch.randn(1_000_00, device=dev)
    code=dump_code(f"cudagraphs_{tag}", elt, x)
    if torch.cuda.is_available():
        f=torch.compile(elt, backend="inductor", fullgraph=True); f(x); torch.cuda.synchronize()
        t=time.time(); 
        for _ in range(200): f(x)
        torch.cuda.synchronize(); print(tag, "elapsed", time.time()-t)
if __name__=="__main__":
    run(False,"off"); run(True,"on")
