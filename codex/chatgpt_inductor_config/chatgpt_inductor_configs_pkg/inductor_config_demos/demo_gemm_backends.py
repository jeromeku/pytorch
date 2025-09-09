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


def mm(a,b): return torch.addmm(torch.zeros(a.size(0), b.size(1), device=a.device, dtype=a.dtype), a,b)
def run(backends, tag):
    torch._inductor.config.max_autotune = True
    torch._inductor.config.max_autotune_gemm_backends = backends
    dev="cuda" if torch.cuda.is_available() else "cpu"
    dt=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    a=torch.randn(512,256,device=dev,dtype=dt); b=torch.randn(256,128,device=dev,dtype=dt)
    code=dump_code(f"gemm_backends_{tag}", mm, a,b)
    print(tag, "triton_" in code, "extern_kernels.addmm" in code)
if __name__=="__main__":
    run("ATEN,TRITON","aten_triton")
    try: run("TRITON","triton_only")
    except Exception as e: print("TRITON-only failed:", e)
