import torch
import torch._inductor.config as inductor_config
from functorch import make_fx
from torch import Tensor
from torch._dynamo.utils import ReinplaceCounters
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops_core
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    subtest,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.logging_utils import logs_to_string

from torch._logging import set_logs
import logging
import torch._inductor.config as inductor_config

inductor_config.force_disable_caches = True

set_logs(inductor=logging.DEBUG)

aten = torch.ops.aten


const = torch.tensor(0.0)
device = GPU_TYPE

@torch.library.custom_op("_reinplacing::sin", mutates_args={"result"})
def sin(x: torch.Tensor, result: torch.Tensor) -> None:
    result.copy_(x.sin())


@torch.library.custom_op("_reinplacing::sin_cos", mutates_args={"out_sin", "out_cos"})
def sin_cos(x: torch.Tensor, out_sin: torch.Tensor, out_cos: torch.Tensor) -> None:
    out_sin.copy_(x.sin())
    out_cos.copy_(x.cos())


import triton  # @manual
import triton.language as tl  # @manual

@triton.jit
def sin_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)

def sin_triton(x, out):
    n_elements = x.numel()
    sin_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)


@torch.library.custom_op("test_view::boo", mutates_args={"x"})
def boo(x: torch.Tensor) -> None:
    x.sin_()

def get_not_inplaced_count(graph):
    counter = 0
    auto_functionalized_found = False
    for node in graph.nodes:
        if (node.target == torch.ops.higher_order.auto_functionalized) or (
            node.target == torch.ops.higher_order.auto_functionalized_v2
        ):
            auto_functionalized_found = True
            counter += len(node.meta["only_clone_these_tensors"])
    assert auto_functionalized_found
    return counter

def test_view_inplaced_functionalize_v2(should_functionalize=True):
    
    def f(arg0_1):
        torch.ops.aten.select.int(arg0_1, 0, 0)
        auto_functionalized = auto_functionalized_v2(
            torch.ops.test_view.boo.default,
            _x_base_index=0,
            _x_size=(3,),
            _x_stride=(1,),
            _x_storage_offset=0,
            _all_bases=[arg0_1],
        )
        getitem_1 = auto_functionalized[1]
        torch.ops.aten.copy_.default(arg0_1, getitem_1)
        return ()

    x1 = torch.randn(3, device=device)
    
    gm = make_fx(f, tracing_mode="fake")(x1)
    reinplace_inplaceable_ops_core(gm.graph)

    not_inplaced_count = get_not_inplaced_count(gm.graph)
    print(not_inplaced_count)

class Tester:
    def _test(self, f):
        nf = torch.compile(f)
        inp = (
            torch.randn(4, device=device),
            torch.ones(2, device=device, dtype=torch.int),
        )
        inp2 = (inp[0].clone(), inp[1].clone())
        
        out_f = f(*inp)
        out_nf = nf(*inp2)
        
        # self.assertEqual(inp, inp2)

def test_dont_modify_input(self):
    def f(x, y):
        return x.index_put((y,), const)

    self._test(f)

def test_should_modify_inner(self):
    def f(x, y):
        x = x.cos()
        x = x.index_put((y,), const)
        return x

    self._test(f)

def test_should_modify_input(self):
    def f(x, y):
        x = x.index_put_((y,), const)
        return x    
    
    self._test(f)


self = Tester()
test_should_modify_input(self)
# test_dont_modify_input(self)