import os
import pytest
import importlib
import torch

def has_gpu():
    return torch.cuda.is_available()

def trtllm_available():
    try:
        importlib.import_module("tensorrt_llm")
        return True
    except Exception:
        return False

@pytest.fixture(scope="session", autouse=True)
def set_logging_env():
    os.environ.setdefault("TORCH_LOGS", "+dynamo,graph,inductor")
    os.environ.setdefault("TORCH_LOGS_FORMAT", "%(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    yield

skip_if_no_gpu = pytest.mark.skipif(not has_gpu(), reason="CUDA GPU required")
skip_if_no_trtllm = pytest.mark.skipif(not trtllm_available(), reason="TensorRT-LLM not importable")

def count_nodes_by_target(gm, substring: str) -> int:
    n = 0
    for node in gm.graph.nodes:
        tgt = str(node.target)
        if substring in tgt:
            n += 1
    return n

def resolve_pass_apply(*module_names):
    """Return (apply_fn, module_name_used) for the first module that exposes a callable `apply`.
    Tries AutoDeploy transform path first (preferred on `main`), then legacy compilation.
    """
    last_err = None
    for modname in module_names:
        try:
            m = importlib.import_module(modname)
            fn = getattr(m, "apply", None)
            if callable(fn):
                return fn, modname
        except Exception as e:
            last_err = e
            continue
    return None, str(last_err or "")
