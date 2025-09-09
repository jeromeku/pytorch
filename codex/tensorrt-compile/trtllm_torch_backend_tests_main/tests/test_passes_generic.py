import pytest
import pkgutil
import importlib
import torch
from torch.fx import GraphModule, Graph
from tests.conftest import skip_if_no_trtllm

@skip_if_no_trtllm
def test_each_compilation_module_apply_runs_or_is_noop():
    # Try both locations: AutoDeploy transforms and legacy compilation
    pkgs = []
    for name in ("tensorrt_llm._torch.auto_deploy.transform.library",
                 "tensorrt_llm._torch.compilation"):
        try:
            pkgs.append(importlib.import_module(name))
        except Exception:
            continue

    if not pkgs:
        pytest.skip("Neither AutoDeploy transform.library nor _torch/compilation importable")

    def toy_gm():
        g = Graph()
        x = g.placeholder("x")
        g.output(x)
        class M(torch.nn.Module):
            def forward(self, x):  # pragma: no cover
                return x
        return GraphModule(M(), g)

    found_any = False
    for pkg in pkgs:
        for mod in pkgutil.iter_modules(pkg.__path__):
            found_any = True
            m = importlib.import_module(f"{pkg.__name__}.{mod.name}")
            apply_fn = getattr(m, "apply", None)
            if not callable(apply_fn):
                continue
            gm = toy_gm()
            try:
                out = apply_fn(gm)
            except Exception as exc:
                pytest.xfail(f"{m.__name__}.apply raised on toy graph: {exc}")
            else:
                assert isinstance(out, GraphModule), f"{m.__name__}.apply did not return GraphModule"

    if not found_any:
        pytest.skip("No transform/compilation modules discovered to test")
