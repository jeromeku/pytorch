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


def ker(x): return torch.nn.functional.gelu(x@x.t())
def run(method, threads, tag):
    torch._inductor.config.worker_start_method=method
    torch._inductor.config.compile_threads=threads
    xs=[torch.randn(256,256) for _ in range(8)]
    t=time.time(); 
    fs=[torch.compile(ker, backend="inductor", fullgraph=True) for _ in xs]
    for f,x in zip(fs,xs): f(x)
    open(os.path.join(ART_DIR,f"workers_{tag}.txt"),"w").write(f"{method=} {threads=} elapsed={time.time()-t:.3f}\n")
    print(tag,"done")
if __name__=="__main__":
    run("fork",2,"fork_2"); run("fork",8,"fork_8")
    try: run("spawn",4,"spawn_4")
    except Exception as e: print("spawn failed:", e)
