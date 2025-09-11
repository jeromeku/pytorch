import torch
from functools import partial

ns = "custom_ops"
fn = "foo"

lib_def = torch.library.Library(ns, "DEF")
lib_meta = torch.library.Library(ns, "IMPL", "Meta")

lib_def.define(f"{fn}(Tensor x, Tensor y) -> Tensor")

def foo_meta(x, y):         # fake-tensor safe
    return x + y

lib_meta.impl(fn, foo_meta)

from torch._inductor.lowering import register_lowering, make_pointwise
from torch._inductor.virtualized import ops

def foo_lowering(x, y):
    fn = partial(ops.inline_asm_elementwise, asm="add.f32 $0, $1, $2;")
    return make_pointwise(fn)(x, y)

register_lowering(getattr(getattr(torch.ops, ns), fn), type_promotion_kind=None)(foo_lowering)
