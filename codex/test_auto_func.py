import torch
import torch._inductor.config as inductor_config
from torch._logging import set_logs
import logging

@torch.library.custom_op("mylib::my_func", mutates_args={"x"})
def my_func(x: torch.Tensor) -> None:
    x[0] = 1

@torch.compile()
def func():
    f = torch.ones(10)

    my_func(f)

    return f

if __name__ == "__main__":
    set_logs(inductor=logging.DEBUG)
    with inductor_config.patch({"enable_auto_functionalized_v2": False}):
        func()
