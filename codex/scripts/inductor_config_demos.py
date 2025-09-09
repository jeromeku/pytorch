#!/usr/bin/env python3
import os
import sys
import textwrap

import torch

from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_triton_code, run_and_get_kernels


def _cuda():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def demo_fx_graph_cache():
    print("== demo_fx_graph_cache ==")
    def f(x):
        return (x.sin() + x.cos()).relu()
    x = torch.randn(8, device=_cuda())
    counters.clear()
    with torch._inductor.config.patch({"fx_graph_cache": True}):
        base = counters["inductor"]["fxgraph_cache_hit"]
        g = torch.compile(f)
        g(x); g(x)
        print("hits+=", counters["inductor"]["fxgraph_cache_hit"] - base)
    with torch._inductor.config.patch({"fx_graph_cache": False}):
        counters.clear()
        g = torch.compile(f)
        g(x); g(x)
        print("hits when disabled:", counters["inductor"]["fxgraph_cache_hit"])


def demo_force_disable_caches():
    print("== demo_force_disable_caches ==")
    def f(x):
        return (x + 1).relu()
    x = torch.randn(4, device=_cuda())
    counters.clear()
    with torch._inductor.config.patch({"fx_graph_cache": True, "force_disable_caches": True}):
        g = torch.compile(f)
        g(x); g(x)
        print("hits:", counters["inductor"]["fxgraph_cache_hit"], "misses:", counters["inductor"]["fxgraph_cache_miss"])  # expect 0 hits


def demo_max_fusion_size():
    print("== demo_max_fusion_size ==")
    def f(x):
        y = x.sin(); z = y * 2 + y.cos(); return z.relu()
    x = torch.randn(1024, device=_cuda())
    with torch._inductor.config.patch({"max_fusion_size": 64}):
        _, k1 = run_and_get_kernels(torch.compile(f), x)
    with torch._inductor.config.patch({"max_fusion_size": 1}):
        _, k2 = run_and_get_kernels(torch.compile(f), x)
    print("kernels: size=64 →", len(k1), "; size=1 →", len(k2))


def demo_unroll_reductions_threshold():
    print("== demo_unroll_reductions_threshold ==")
    def reduce_small(x):
        return x.sum(dim=-1)
    x = torch.randn(32, 4, device=_cuda())
    with torch._inductor.config.patch({"unroll_reductions_threshold": 1}):
        code_small = run_and_get_triton_code(torch.compile(reduce_small), x)
    with torch._inductor.config.patch({"unroll_reductions_threshold": 128}):
        code_large = run_and_get_triton_code(torch.compile(reduce_small), x)
    print("'reduce' present small?", "reduce" in code_small, " large?", "reduce" in code_large)


def demo_use_block_ptr():
    print("== demo_use_block_ptr ==")
    def f(a, b):
        return (a @ b).relu()
    a = torch.randn(128, 64, device=_cuda()); b = torch.randn(64, 128, device=_cuda())
    with torch._inductor.config.patch({"triton.use_block_ptr": False}):
        code0 = run_and_get_triton_code(torch.compile(f), a, b)
    with torch._inductor.config.patch({"triton.use_block_ptr": True}):
        code1 = run_and_get_triton_code(torch.compile(f), a, b)
    print("tl.int64 in code0?", "tl.int64" in code0, "; in code1?", "tl.int64" in code1)


def demo_size_asserts():
    print("== demo_size_asserts ==")
    def f(x): return x.view(x.shape)
    x = torch.randn(32, 32, device=_cuda())
    with torch._inductor.config.patch({"size_asserts": True}):
        code_on = run_and_get_triton_code(torch.compile(f), x)
    with torch._inductor.config.patch({"size_asserts": False}):
        code_off = run_and_get_triton_code(torch.compile(f), x)
    print("assert_size_stride present?", "assert_size_stride" in code_on, "; off?", "assert_size_stride" in code_off)


def demo_autotune_at_compile_time():
    print("== demo_autotune_at_compile_time ==")
    def f(x): return (x * x).relu()
    x = torch.randn(1 << 20, device=_cuda())
    with torch._inductor.config.patch({"triton.autotune_at_compile_time": False}):
        code_defer = run_and_get_triton_code(torch.compile(f), x)
    with torch._inductor.config.patch({"triton.autotune_at_compile_time": True}):
        code_ct = run_and_get_triton_code(torch.compile(f), x)
    print("len(defer)=", len(code_defer), " len(ct)=", len(code_ct))


def demo_joint_graph_constant_folding():
    print("== demo_joint_graph_constant_folding ==")
    # Capture post-grad logs and count nodes as a simple heuristic
    from torch.testing._internal.logging_utils import logs_to_string
    def f(x):
        y = x * 2
        z = y + torch.ones_like(y)  # constant can fold
        return z
    x = torch.randn(32, device=_cuda())
    def run_and_count(cf):
        log_stream, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
        with ctx():
            torch.compile(f)(x)
        s = log_stream.getvalue()
        return s.count("placeholder") + s.count("call_function") + s.count("output")
    with torch._inductor.config.patch({"joint_graph_constant_folding": True}):
        n1 = run_and_count(True)
    with torch._inductor.config.patch({"joint_graph_constant_folding": False}):
        n2 = run_and_count(False)
    print("node count folded:", n1, " vs unfurled:", n2)


def demo_cudagraphs():
    print("== demo_cudagraphs ==")
    def f(x): return (x.sin() + 1.0).cos()
    x = torch.randn(8192, device=_cuda())
    with torch._inductor.config.patch({"triton.cudagraphs": True}):
        g = torch.compile(f)
        g(x); g(x)
        print("ran with cudagraphs=True")
    with torch._inductor.config.patch({"triton.cudagraphs": False}):
        g2 = torch.compile(f)
        g2(x); g2(x)
        print("ran with cudagraphs=False")


def main():
    demos = [
        demo_fx_graph_cache,
        demo_force_disable_caches,
        demo_max_fusion_size,
        demo_unroll_reductions_threshold,
        demo_use_block_ptr,
        demo_size_asserts,
        demo_autotune_at_compile_time,
        demo_joint_graph_constant_folding,
        demo_cudagraphs,
    ]
    for d in demos:
        try:
            d()
        except Exception as e:
            print(f"[warn] {d.__name__} failed: {e}")


if __name__ == "__main__":
    main()

