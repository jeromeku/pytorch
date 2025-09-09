from torch.testing._internal.logging_utils import logs_to_string, multiple_logs_to_string
#    multiple_logs_to_string("torch._inductor.compile_fx", "pre_grad_graphs", "post_grad_graphs")
import torch

@torch.library.custom_op("mylib::my_func", mutates_args={"x"})
def my_func(x: torch.Tensor) -> None:
    x[0] = 1

@torch.compile()
def func():
    f = torch.ones(10)

    my_func(f)

    return f

# log_stream, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")

# with ctx():
func()
# print(log_stream.getvalue().strip())

# def get_post_grad_graph(f, inputs):
#     log_stream, ctx = logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")
#     with ctx():
#         f(*inputs)
#     post_grad_graph = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()
#     return post_grad_graph

# def test_remove_noop_slice1(device="cuda"):
#     def f(x):
#         x = x + 1
#         y = torch.ops.aten.slice(x, -1, 0, -1)  # not a noop
#         return y + 1

#     f = torch.compile(f)
#     x = torch.ones((2, 3, 2), device=device)
#     torch._dynamo.mark_dynamic(x, 0)
#     torch._dynamo.mark_dynamic(x, 1)
#     post_grad_graph = get_post_grad_graph(f, (x,))
#     print(post_grad_graph)

#     expected_graph = f"""\
# def forward(self, arg0_1: "Sym(s77)", arg1_1: "Sym(s27)", arg2_1: "f32[s77, s27, 2][2*s27, 2, 1]{str(x.device)}"):
#     add: "f32[s77, s27, 2][2*s27, 2, 1]{str(x.device)}" = torch.ops.aten.add.Tensor(arg2_1, 1);  arg2_1 = None
#     slice_1: "f32[s77, s27, 1][2*s27, 2, 1]{str(x.device)}" = torch.ops.aten.slice.Tensor(add, -1, 0, -1);  add = None
#     add_9: "f32[s77, s27, 1][s27, 1, 1]{str(x.device)}" = torch.ops.aten.add.Tensor(slice_1, 1);  slice_1 = None
#     return (add_9,)"""  # noqa: B950

# test_remove_noop_slice1()