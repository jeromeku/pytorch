# Owner(s): ["module: inductor"]
import contextlib
import re
from unittest.mock import patch

import functorch
import torch
import torch._inductor.config as config
import torch.autograd
from torch._inductor import metrics
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code

########################
# Explanation of Tests #
########################
# These tests are all testing *memory accesses* of TorchInductor.
# They are intended to be deterministic performance tests.
# The expect tests are all measuring the number of memory bytes read/written by
# the code that Inductor has generated
#
# If the test is failing because the number became smaller, feel free to lower it.
# On the other hand, if the test is failing because the number became larger,
# that means that your change is leading to *more* memory accesses on this test.
#
# That may still be aceeptable, but be aware that you are likely lowering
# performance for that setting.
#
# Defines all the kernels for tests
from torch.testing._internal.triton_utils import (
    HAS_CUDA_AND_TRITON,
    requires_cuda_and_triton,
)


# set so that metrics appear
torch._logging.set_logs(inductor_metrics=True)

if HAS_CUDA_AND_TRITON:
    import triton  # @manual
    import triton.language as tl  # @manual

    from torch.testing._internal.triton_utils import add_kernel

aten = torch.ops.aten


def compile_but_use_eager(gm, example_inputs):
    def inner_compile(gm, *args, **kwargs):
        compile_fx_inner(gm, *args, **kwargs)
        return gm

    return compile_fx(gm, example_inputs, inner_compile=inner_compile)


def count_numel(f, *args):
    """
    Assumes all inputs are fp32
    """
    metrics.reset()
    torch.compile(f, backend=compile_but_use_eager)(*args)
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)


def count_numel_train(f, *args):
    """
    Assumes all inputs are fp32
    """
    metrics.reset()

    f = torch.compile(f, backend=compile_but_use_eager)
    out = f(*args)
    res = 0
    for o in out:
        res += o.mean()
    res.backward()
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)


DEVICE = "cuda"


def T(*size, dtype=torch.float32, device=DEVICE, grad=False):
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)


def TI(*size, mx=10, dtype=torch.int32, device=DEVICE):
    return torch.randint(0, mx, size, dtype=dtype, device=device)


class TestCase(InductorTestCase):
    device = DEVICE

class InplacingTests(TestCase):

    def test_inplace_custom_op_training_two_mutated_inputs(self):
        @torch.library.custom_op(
            "_reinplacing::sin_cos", mutates_args={"out_sin", "out_cos"}
        )
        def sin_cos(
            x: torch.Tensor, out_sin: torch.Tensor, out_cos: torch.Tensor
        ) -> None:
            out_sin.copy_(x.sin())
            out_cos.copy_(x.cos())

        def f(x):
            out0 = torch.empty_like(x)
            out1 = torch.empty_like(x)
            sin_cos(x, out0, out1)
            return x.clone(), out0, out1

        x = T(3, grad=True)
        self.assertExpectedInline(count_numel(f, x), """21""")

    def test_inplace_custom_op_training(self):
        @torch.library.custom_op("_reinplacing::sin", mutates_args={"result"})
        def sin(x: torch.Tensor, result: torch.Tensor) -> None:
            result.copy_(x.sin())

        factory_op = torch.empty_like

        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = factory_op(x)
                sin(x, out)
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad):
                (saved,) = ctx.saved_tensors
                out = factory_op(grad)
                sin(saved, out)
                return out

        def f(x):
            return MySin.apply(x)

        x = T(3, grad=True)
        self.assertExpectedInline(count_numel_train(f, x), """9""")

    def test_inplace_custom_op(self, disable_functionalize_v2: bool = False):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x, Tensor(a!) out) -> ()")

            def foo(x: torch.Tensor, out: torch.Tensor) -> None:
                out.copy_(x.sin())

            m.impl("foo", foo, "CompositeExplicitAutograd")

            def f(x, out):
                torch.ops.mylib.foo(x, out)
                return out

            def f2(x, out):
                torch.ops.mylib.foo(x, out)
                torch.ops.mylib.foo(out, out)
                return out

            x = T(3)
            out = T(3)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, fullgraph=True), x, out
            )
            # print(f"f code:\n{code}")
            # compiled_out, (code3,) = run_and_get_code(
            #     torch.compile(f2, fullgraph=True), x, out
            # )
            # print(f"f2 code:\n{code}")
            
            # self.assertEqual(compiled_out, x.sin().sin().sin())

            # Check that we are allocating the minimum number of intermediate buffers
            matches = re.findall(r"empty_strided_\w+\(", code)
            print(f"{matches=}")
            self.assertEqual(len(matches), 0)

            # self.assertExpectedInline(count_numel(f, x, out), """21""")

    def test_inplace_custom_op_intermediate(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x, Tensor(a!) out) -> ()")

            def foo(x: torch.Tensor, out: torch.Tensor) -> None:
                out.copy_(x.sin())

            m.impl("foo", foo, "CompositeExplicitAutograd")

            def f(x, out):
                out = torch.empty_like(x)
                torch.ops.mylib.foo(x, out)
                torch.ops.mylib.foo(out, out)
                torch.ops.mylib.foo(out, out)
                return out

            x = T(3)
            out = T(3)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, fullgraph=True), x, out
            )
            self.assertEqual(compiled_out, x.sin().sin().sin())

            # Check that we are allocating the minimum number of intermediate buffers
            matches = re.findall(r"empty_strided_\w+\(", code)
            self.assertEqual(len(matches), 1)

            self.assertExpectedInline(count_numel(f, x, out), """21""")

    def test_inplace_custom_op_two_mutated_inputs(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor q, Tensor(a!) k_cache, Tensor(b!) v_cache) -> Tensor")

            def foo(q, k_cache, v_cache):
                k_cache.add_(1)
                v_cache.add_(1)
                return q + 1

            m.impl("foo", foo, "CompositeExplicitAutograd")

            q = T(3)
            k_cache = T(3)
            v_cache = torch.rand_like(k_cache)

            def f():
                x = 0
                for _ in range(2):
                    x = x + torch.ops.mylib.foo(q, k_cache, v_cache)
                return x

            _, (code,) = run_and_get_code(
                torch.compile(f, fullgraph=True),
            )

            # Check that we are allocating the minimum number of intermediate buffers
            matches = re.findall(r"empty_strided_\w+\(", code)
            self.assertEqual(len(matches), 1)

            self.assertExpectedInline(count_numel(f), """39""")

import logging
from torch._inductor import config

torch._logging.set_logs(inductor=logging.DEBUG)
tests = InplacingTests()

config.enable_auto_functionalized_v2 = True
# config.trace.enabled = True
config.force_disable_caches = True

tests.test_inplace_custom_op(disable_functionalize_v2=True)