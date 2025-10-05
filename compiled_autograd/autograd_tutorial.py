import torch
from torch._logging import set_logs
from torch._inductor import config as inductor_config
from torch._dynamo import config as dynamo_config
from contextlib import redirect_stderr, redirect_stdout, ExitStack
import sys
import logging
import torch._logging._internal as torch_logging


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10, bias=False)

    def forward(self, x):
        return self.linear(x)


def main():
    model = Model()
    x = torch.randn(10)

    torch._dynamo.config.compiled_autograd = COMPILED_AG

    @torch.compile
    def train(model, x):
        loss = model(x).sum()
        loss.backward()

    train(model, x)


if __name__ == "__main__":
    LOG_COMPILED_AG = False
    COMPILED_AG = True
    AOT_GRAPH = True
    AOT_LOGGING = True
    DYN_LOGGING = False
    LOG_FORMAT = "[%(levelname)s] %(process)d %(pathname)s:%(lineno)d — %(message)s"
    torch_logging.DEFAULT_FORMATTER = logging.Formatter(LOG_FORMAT)
    torch_logging._init_logs()

    inductor_config.force_disable_caches = True

    set_logs(
        aot=logging.DEBUG if AOT_LOGGING else None,
        dynamo=logging.debug if DYN_LOGGING else None,
        compiled_autograd_verbose=LOG_COMPILED_AG,
        aot_joint_graph=AOT_GRAPH,
    )

    with ExitStack() as stack:
        _stderr = stack.enter_context(open("err.log", "w"))
        _stdout = stack.enter_context(open("out.log", "w"))
        stack.enter_context(redirect_stderr(_stderr))
        stack.enter_context(redirect_stdout(_stdout))
        print("TESTING")
        print("TESTING ERR", file=sys.stderr)
        main()

    main()
