#!/usr/bin/env python3
import argparse
import torch

from torch._inductor.graph import GraphLowering
from torch._inductor.debug import DebugFormatter
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V


def demo_fn(x: torch.Tensor) -> torch.Tensor:
    # Elementwise + matmul to exercise multiple IR nodes
    return (x @ x.t()).relu() + x.sin()


def print_ir(nodes, title: str):
    try:
        # Use the internal IR pretty-printer
        s = DebugFormatter(None)._write_ir(nodes)  # type: ignore[arg-type]
        print(f"\n=== {title} ===\n{s}")
    except Exception:
        # Fallback: just list node names/types
        print(f"\n=== {title} (fallback) ===")
        for n in nodes:
            name = getattr(getattr(n, "node", None), "__class__", type(n)).__name__
            print("-", getattr(n, "get_name", lambda: str(n))(), "::", name)


def main():
    p = argparse.ArgumentParser(description="Lower to Inductor IR and show pre/post fusion snapshots")
    p.add_argument("--shape", default="64,64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    args = p.parse_args()

    shape = tuple(int(s) for s in args.shape.split(","))
    dtype = getattr(torch, args.dtype)
    x = torch.randn(*shape, device=args.device, dtype=dtype)

    from torch.fx.experimental.proxy_tensor import make_fx
    gm = make_fx(demo_fn)(x)

    # Lower FX -> Inductor IR
    gl = GraphLowering(gm, example_inputs=[x], shape_env=None, graph_id=0, cpp_wrapper=False,
                       aot_mode=False, extern_node_serializer=None, is_inference=True,
                       is_backward=False, const_output_index=None, const_wrapper_code=None,
                       const_kernel_code=None, const_module=None, inputs_to_check=None, fx_wrapper=False)
    with V.set_graph_handler(gl), V.set_extern_kernel_nodes([]):
        gl.run(x)

    # Pre-fusion operations list
    ops = gl.operations
    print_ir(ops, "Inductor IR BEFORE fusion (operations)")

    # Build a Scheduler (this performs fusion and ordering)
    sched = Scheduler(ops)
    print_ir(sched.nodes, "Inductor IR AFTER fusion (scheduler nodes)")

    # Optionally, emit code to ensure codegen works (stdout summary only)
    try:
        wrapper, kernels = gl.codegen()
        print("\n=== Codegen summary ===")
        print("wrapper lines:", len(wrapper.value.splitlines()))
        print("kernels lines:", len(kernels.value.splitlines()))
    except Exception as e:
        print("[warn] codegen failed:", e)


if __name__ == "__main__":
    main()

