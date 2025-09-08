import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops_core
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
from functorch import make_fx

# 1) A tiny mutating custom op: mylib::inc_(Tensor(a!) x) -> ()
@torch.library.custom_op("mylib::inc_", mutates_args={"x"})
def inc_(x: torch.Tensor) -> None:
    x.add_(1)

# 2) A function that uses auto_functionalized_v2 around the mutating op.
#    We: (a) mark x as a base (index 0) and (b) list all bases.
def f(x):
    # v2 returns: (op_outputs..., bases_after_mutation...)
    outs = auto_functionalized_v2(
        torch.ops.mylib.inc_.default,
        _x_base_index=0,       # x is a base (no view); no size/stride keys means NotView
        _all_bases=[x],
    )
    new_x = outs[1]           # first base result
    torch.ops.aten.copy_.default(x, new_x)  # epilogue writeback for input mutation
    return x

# 3) Trace and print BEFORE any inductor post-grad passes
x = torch.randn(3)
gm = make_fx(f, tracing_mode="fake")(x)
print("=== BEFORE ===")
print(gm.graph)

# 4) Run the reinplace analysis (decide which bases must be cloned)
reinplace_inplaceable_ops_core(gm.graph)

# 5) Decompose the HOP into actual ops + (possibly) clones per the metadata
decompose_auto_functionalized(gm.graph)
gm.graph.eliminate_dead_code()
gm.recompile()

print("=== AFTER ===")
print(gm.graph)
