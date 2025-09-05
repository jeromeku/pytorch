#!/usr/bin/env python3
import argparse
import torch

from torch._inductor.fx_passes.pre_grad import pre_grad_passes
from torch._inductor.fx_passes.joint_graph import joint_graph_passes
from torch._inductor.fx_passes.post_grad import post_grad_passes


def default_fn(x: torch.Tensor) -> torch.Tensor:
    # Simple chain with reshapes and pointwise to exercise FX passes
    y = x.view(-1).view(x.shape)
    y = torch.sin(y)
    y = torch.relu(y)
    return y


def print_graph(tag: str, gm: torch.fx.GraphModule) -> None:
    print(f"\n=== {tag} ===")
    print(gm.print_readable(print_output=False, include_stride=True, include_device=True))


def main():
    p = argparse.ArgumentParser(description="Run Inductor FX passes and show before/after graphs")
    p.add_argument("--stage", choices=["pre", "joint", "post"], default="pre")
    p.add_argument("--shape", default="8,8")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    shape = tuple(int(s) for s in args.shape.split(","))
    x = torch.randn(*shape, device=args.device)

    # Build a small FX graph via make_fx
    from torch.fx.experimental.proxy_tensor import make_fx

    gm = make_fx(default_fn)(x)
    print_graph("Initial FX", gm)

    if args.stage == "pre":
        gm2 = pre_grad_passes(gm, example_inputs=(x,))
        print_graph("After pre_grad_passes", gm2)
    elif args.stage == "joint":
        joint_graph_passes(gm)
        print_graph("After joint_graph_passes", gm)
    else:
        # Post-grad expects a normalized/functional graph; for demo we re-use gm
        post_grad_passes(gm, is_inference=True)
        print_graph("After post_grad_passes", gm)


if __name__ == "__main__":
    main()

