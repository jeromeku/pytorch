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


def tiny(x): return (x+1).t().contiguous()
def run(enabled, tag):
    try:
        import torch._logging as tlog; tlog.set_logs(output_code=True, graph=True)
    except Exception: pass
    torch._inductor.config.trace.enabled = enabled
    x=torch.randn(128,128); code=dump_code(f"trace_{tag}", tiny, x)
    print(tag, any(s in code for s in ["# scheduler","GraphModule","call_function"]))
if __name__=="__main__":
    run(False,"off"); run(True,"on")
