import torch

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.utils.benchmark as benchmark
from dataclasses import dataclass, field
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE as FLEX_ATTENTION_BLOCK_SIZE,
)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from flex_utils import create_causal_block_mask_fast, BlockMask
from contextlib import contextmanager
from functools import partial

DEVICE = "cuda"
DTYPE = torch.bfloat16


# Lets define the hyper-parameters of our input
@dataclass
class ModelConfig:
    bsz: int = 4
    qlen: int = 2048
    num_heads: int = 32
    hidden_size: int = 4096
    head_dim: int = 128
    num_key_value_heads: int = 8
    num_key_value_groups: int = field(init=False)

    def __post_init__(self):
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


@torch.inference_mode
def assert_close(old, new, msg="", dtype: torch.dtype = DTYPE):
    atol, rtol = {
        torch.float16: [0.5, 10000],
        torch.bfloat16: [0.5, 10000],
    }[dtype]

    if old is not None and new is not None:
        difference = torch.amax(old - new)
        torch._assert(difference <= atol, msg(f"difference = {difference}"))


def make_inputs(model_config: ModelConfig, device: str = DEVICE, dtype: torch.dtype = DTYPE):
    torch.cuda.manual_seed(3407)
    bsz, qlen, hidden_size = model_config.bsz, model_config.qlen, model_config.hidden_size
    num_heads, head_dim, num_key_value_heads, num_key_value_groups = (
        model_config.num_heads,
        model_config.head_dim,
        model_config.num_key_value_heads,
        model_config.num_key_value_groups,
    )
    query = torch.rand(
        model_config.bsz, model_config.qlen, model_config.hidden_size, device=device, dtype=dtype
    )
    key = torch.rand(bsz, qlen, hidden_size // num_key_value_groups, device=device, dtype=dtype)
    value = torch.rand(bsz, qlen, hidden_size // num_key_value_groups, device=device, dtype=dtype)

    query = query.view(bsz, qlen, num_heads, head_dim)
    key = key.view(bsz, qlen, num_key_value_heads, head_dim)
    value = value.view(bsz, qlen, num_key_value_heads, head_dim)

    return query, key, value


def causal_sdpa(num_key_value_groups, query, key, value, backwards: bool = False):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=num_key_value_groups != 1
    )
    out = out.transpose(1, 2).contiguous()
    if backwards:
        out.backward(torch.ones_like(out))


def benchmark_sdpa(
    model_config: ModelConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend: SDPBackend,  # type: ignore
):
    with sdpa_kernel(backend):
        t_us = benchmark_torch_function_in_microseconds(
            causal_sdpa, model_config.num_key_value_groups, query, key, value
        )

    return t_us


def configure_flex_attn(
    model_config: ModelConfig,
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    compile_config: dict = None,
):
    block_mask = create_causal_block_mask_fast(
        batch_size=batch_size,
        num_heads=model_config.num_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
    )
    flex_attention_compiled = torch.compile(flex_attention, **compile_config)
    return block_mask, flex_attention_compiled


def run_flex_attn(
    flex_attn: callable,
    block_mask: BlockMask,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    backwards: bool = False,
    enable_gqa: bool = True,
):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    out = flex_attn(
        query,
        key,
        value,
        score_mod=None,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    )
    out = out.transpose(1, 2).contiguous()
    if backwards:
        out.backward(torch.ones_like(out))


def benchmark_flex_attn(
    model_config: ModelConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compile_config: dict = None,  # type: ignore,
    backwards: bool = False,
):
    block_mask, flex_attn_func = configure_flex_attn(
        model_config, query, key, value, compile_config=compile_config
    )
    enable_gqa = model_config.num_key_value_groups > 1
    flex_attn_func = partial(flex_attn_func, enable_gqa=enable_gqa, backwards=backwards)
    t_us = benchmark_torch_function_in_microseconds(
        run_flex_attn, flex_attn_func, block_mask, query, key, value
    )
    return t_us


def clone_inputs(*args):
    return [t.detach().clone() for t in args]


class BenchmarkContext:
    def __init__(self, query, key, value, tag: str = ""):
        self.query, self.key, self.value = query, key, value
        self.tag = tag

    def __enter__(self):
        print(f"Benchmarking {self.tag}")
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def get_inputs(self):
        return clone_inputs(self.query, self.key, self.value)

    def set_tag(self, tag: str):
        self.tag = tag

    def set_benchmark_fn(self, fn: callable, *args, **kwargs):
        return 
def run_benchmarks(
    model_config: ModelConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sdpbackend: SDPBackend = SDPBackend.FLASH_ATTENTION
):

    q, k, v = clone_inputs(query, key, value)
    t_us = benchmark_sdpa(model_config, q, k, v, backend=sdpbackend)
    print(f"SDPA attn with {sdpbackend}: {t_us:.2f}us")

if __name__ == "__main__":
    config = ModelConfig()
    query, key, value = make_inputs(config, device=DEVICE, dtype=DTYPE)
    run_benchmarks(config, query, key, value)