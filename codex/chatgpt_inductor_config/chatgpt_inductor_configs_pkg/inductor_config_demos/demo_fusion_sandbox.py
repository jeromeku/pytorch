#!/usr/bin/env python3
"""
Fusion Sandbox: toggle Inductor fusion heuristics and compare outcomes.

- Captures the scheduler's "fusion" logger output
  (see torch/_inductor/scheduler.py:79)
- Counts emitted kernels (proxy for fusion strength)
- Lets you switch heuristics like:
  - max_fusion_size (torch/_inductor/config.py:658)
  - score_fusion_memory_threshold (torch/_inductor/config.py:647)
  - realize_acc_reads_size_threshold (torch/_inductor/config.py:613)
  - pick_loop_orders (torch/_inductor/scheduler.py:2002–2046 uses it)

Example:
  python demo_fusion_sandbox.py --model pointwise \
    --baseline max_fusion_size=64 \
    --variant max_fusion_size=1,score_fusion_memory_threshold=1000,realize_acc_reads_size_threshold=1024,pick_loop_orders=False

Models:
  - pointwise: small chain with shared reads
  - mmrelu: GEMM + ReLU (epilogue fusion candidate)
  - index_mismatch: defeats horizontal fusion via indexing change
"""

import argparse
import ast
import sys
from typing import Any, Dict, Tuple

import torch
from torch.testing._internal.logging_utils import logs_to_string
from torch._inductor.utils import run_and_get_kernels


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def model_pointwise(x: torch.Tensor) -> torch.Tensor:
    y = x * 2
    z = y + x
    return z.relu()


def model_mmrelu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a @ b).relu()


def model_index_mismatch(x: torch.Tensor) -> torch.Tensor:
    y = x.roll(1, dims=0)
    return y + x


def parse_patches(s: str) -> Dict[str, Any]:
    # Accept comma-separated key=val; bools/nones parsed; ints auto-cast
    if not s:
        return {}
    out: Dict[str, Any] = {}
    for kv in s.split(","):
        if not kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            parsed = ast.literal_eval(v)
        except Exception:
            parsed = v
        out[k] = parsed
    return out


def _classify_backend(kernels_src: list[str]) -> str:
    if not kernels_src:
        return "none"
    txt = "\n".join(kernels_src).lower()
    if "tl.load" in txt or "@triton" in txt or "triton_" in txt:
        return "triton"
    if "extern \"c\"" in txt or "__global__" in txt or "__device__" in txt:
        return "cuda-cpp"
    if "at::" in txt or "c10::" in txt or "aten::" in txt:
        return "cpp"
    return "unknown"


def run_case(model: str, patches: Dict[str, Any]) -> Tuple[int, str, str]:
    dev = _device()
    if model == "pointwise":
        fn = lambda: torch.compile(model_pointwise)(torch.randn(1 << 18, device=dev))
    elif model == "mmrelu":
        a = torch.randn(512, 512, device=dev)
        b = torch.randn(512, 512, device=dev)
        fn = lambda: torch.compile(model_mmrelu)(a, b)
    elif model == "index_mismatch":
        fn = lambda: torch.compile(model_index_mismatch)(torch.randn(1 << 18, device=dev))
    else:
        raise SystemExit(f"unknown --model {model}")

    log, ctx = logs_to_string("torch._inductor.scheduler", "fusion")
    with torch._inductor.config.patch(patches), ctx():
        # also gather kernel count
        if model == "mmrelu":
            a = torch.randn(512, 512, device=dev)
            b = torch.randn(512, 512, device=dev)
            _, kernels = run_and_get_kernels(torch.compile(model_mmrelu), a, b)
        else:
            x = torch.randn(1 << 18, device=dev)
            _, kernels = run_and_get_kernels(torch.compile(model_pointwise if model=="pointwise" else model_index_mismatch), x)
    backend = _classify_backend(kernels)
    return len(kernels), backend, log.getvalue()


def summarize(tag: str, kernels: int, backend: str, log_text: str) -> None:
    print(f"\n== {tag} ==")
    print(f"kernels: {kernels} | backend: {backend}")
    lines = log_text.strip().splitlines()
    head = "\n".join(lines[:80])
    print("-- fusion log (first 80 lines) --\n" + head)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["pointwise", "mmrelu", "index_mismatch"], default="pointwise")
    p.add_argument("--baseline", default="max_fusion_size=64")
    p.add_argument(
        "--variant",
        default=(
            "max_fusion_size=1,score_fusion_memory_threshold=1000,"
            "realize_acc_reads_size_threshold=1024,pick_loop_orders=False"
        ),
    )
    p.add_argument("--variants", default="", help="Optional '|' separated list of variant patch strings")
    p.add_argument("--csv", default="", help="Optional CSV output path for results")
    args = p.parse_args(argv)

    base_patches = parse_patches(args.baseline)
    var_patches = parse_patches(args.variant)

    print("device:", _device())
    print("baseline patches:", base_patches)
    print("variant patches:", var_patches)

    series = [("BASELINE", base_patches)]
    if args.variant:
        series.append(("VARIANT", var_patches))
    if args.variants:
        for i, s in enumerate(args.variants.split("|")):
            s = s.strip()
            if not s:
                continue
            series.append((f"VARIANT{i+1}", parse_patches(s)))

    results = []
    for tag, patches in series:
        kcnt, backend, flog = run_case(args.model, patches)
        summarize(tag, kcnt, backend, flog)
        results.append((tag, patches, kcnt, backend))

    if args.csv:
        import csv, json
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tag", "model", "device", "patches", "kernels", "backend"])
            dev = _device()
            for tag, patches, kcnt, backend in results:
                w.writerow([tag, args.model, dev, json.dumps(patches, sort_keys=True), kcnt, backend])
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
