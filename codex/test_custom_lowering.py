# Owner(s): ["module: inductor"]

from functools import partial
from unittest import skipIf

import torch
from torch._inductor.ir import Pointwise
from torch._inductor.lowering import make_pointwise, register_lowering
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import ops
from torch.testing._internal.common_utils import skipIfRocm, skipIfXpu
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)

class TestCustomLowering(InductorTestCase):
    def __init__(self):
        self.test_inductor_ops = torch.library.Library(  # noqa: TOR901
            "test_inductor_ops", "DEF"
        )
        self.device_list = ["Meta", "CUDA"]
        for device in self.device_list:
            setattr(
                self,
                "impl_" + device.lower(),
                torch.library.Library(  # noqa: TOR901
                    "test_inductor_ops", "IMPL", device
                ),
            )
        breakpoint()
        self._register_asm_op()

    def _register_asm_op(cls):
        # Approximation of fbgemm.jagged_to_padded_dense_forward
        cls.test_inductor_ops.define("tanh_approx(Tensor input) -> Tensor")

        def tanh_approx_meta(inp):
            return torch.tanh(inp)

        cls.impl_meta.impl("tanh_approx", tanh_approx_meta)

        def tanh_approx_lowering(inp):
            fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
            return make_pointwise(fn)(inp)

        register_lowering(
            torch.ops.test_inductor_ops.tanh_approx, type_promotion_kind=None
        )(tanh_approx_lowering)

        cls.test_inductor_ops.define("add_custom(Tensor a, Tensor b) -> Tensor")

        def add_custom(a, b):
            return a + b

        cls.impl_meta.impl("add_custom", add_custom)

        def add_custom_lowering(a, b):
            fn = partial(ops.inline_asm_elementwise, asm="add.f32 $0, $1, $2;")
            return make_pointwise(fn)(a, b)

        register_lowering(
            torch.ops.test_inductor_ops.add_custom, type_promotion_kind=None
        )(add_custom_lowering)

    def test_tanh_approx(self):
        def fn(inp):
            return torch.ops.test_inductor_ops.tanh_approx(inp)

        inp = torch.randn(32, device=GPU_TYPE)
        fn_opt = torch.compile(fn)

        a = torch.tanh(inp)
        b = fn_opt(inp)
        self.assertEqual(a, b)

    def test_multi_inp_asm(self):
        def fn(a, b):
            return torch.ops.test_inductor_ops.add_custom(a, b)

        a = torch.randn(32, device=GPU_TYPE)
        b = torch.randn(32, device=GPU_TYPE)
        fn_opt = torch.compile(fn)

        out1 = a + b
        out2 = fn_opt(a, b)
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    tester = TestCustomLowering()
    tester.test_tanh_approx()