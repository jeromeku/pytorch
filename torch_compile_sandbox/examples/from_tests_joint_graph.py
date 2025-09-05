import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.fx_passes import joint_graph


def f(x):
    # Based on test_pattern_matcher pointless view pair example
    x = torch.ops.aten.view.default(x, [3, 5, 7])
    x = torch.ops.aten.view.default(x, [-1, 7])
    return x


if __name__ == "__main__":
    x = torch.randn(105)
    gm = make_fx(f)(x)
    print("Before:")
    print(gm.print_readable(print_output=False))
    joint_graph.joint_graph_passes(gm)
    print("\nAfter joint_graph_passes:")
    print(gm.print_readable(print_output=False))

