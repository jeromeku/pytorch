import argparse
import importlib
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch._inductor.pattern_matcher import PatternPrettyPrinter

# Prefer AutoDeploy transform on main
PASS_ALIASES = {
    "fuse_norm": [
        "tensorrt_llm._torch.auto_deploy.transform.library.rms_norm",
        "tensorrt_llm._torch.compilation.fuse_norm",
        "tensorrt_llm._torch.compilation.norm_fusion",
    ],
    "fix_functionalization": [
        "tensorrt_llm._torch.auto_deploy.transform.library.functionalization",
        "tensorrt_llm._torch.compilation.fix_functionalization",
    ],
}

def print_gm(gm, title):
    print(f"\n===== {title} =====")
    PatternPrettyPrinter().print_graph(gm.graph)

def resolve_apply(pass_name):
    mods = PASS_ALIASES.get(pass_name, [pass_name])
    last = None
    for modname in mods:
        try:
            m = importlib.import_module(modname)
            fn = getattr(m, "apply", None)
            if callable(fn):
                return fn, modname
        except Exception as e:
            last = e
    raise RuntimeError(f"No callable 'apply' for pass {pass_name}; last error: {last}")

def build_add_rms_model():
    RMSNorm = importlib.import_module("tensorrt_llm._torch.modules.rms_norm").RMSNorm
    class AddRms(nn.Module):
        def __init__(self, hidden=64, eps=1e-5, dtype=torch.float16):
            super().__init__()
            self.norm = RMSNorm(hidden_size=hidden, eps=eps, dtype=dtype)
        def forward(self, x, r):
            return self.norm(x + r)
    return AddRms().eval().cuda()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_name", required=True)
    args = ap.parse_args()

    apply_fn, used = resolve_apply(args.pass_name)
    print(f"Using pass from: {used}")

    model = build_add_rms_model()
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    r = torch.randn(8, 64, device="cuda", dtype=torch.float16)

    gm = symbolic_trace(model)
    print_gm(gm, "BEFORE")
    gm2 = apply_fn(gm)
    print_gm(gm2, "AFTER")

if __name__ == "__main__":
    main()
