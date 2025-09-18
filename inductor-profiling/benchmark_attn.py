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


def causal_sdpa(num_key_value_groups, query, key, value, *, enable_gqa: bool, backwards: bool):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=enable_gqa)
    out = out.transpose(1, 2).contiguous()
    if backwards:
        out.backward(torch.ones_like(out))


def benchmark_sdpa(
    model_config: ModelConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend: SDPBackend,  # type: ignore
    backwards: bool = False,
):
    _sdpa = partial(
        causal_sdpa, enable_gqa=model_config.num_key_value_groups > 1, backwards=backwards
    )
    with sdpa_kernel(backend):
        t_us = benchmark_torch_function_in_microseconds(
            _sdpa, model_config.num_key_value_groups, query, key, value
        )

    return t_us


def configure_flex_attn(
    *,
    q_seq_len: int,
    kv_seq_len: int,
    batch_size: int = None,
    num_heads: int = None,
    compile_config: dict = None,
    device: str = DEVICE,
    separate_full_blocks: bool = True,
    compile_block_mask_fn: bool = True
):
    if compile_block_mask_fn:
        _block_mask_fn = torch.compile(create_causal_block_mask_fast)
    else: 
        _block_mask_fn = create_causal_block_mask_fast
    block_mask = _block_mask_fn(
        batch_size=batch_size,
        num_heads=num_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        device=device,
        separate_full_blocks=separate_full_blocks
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
    mask_batch_size: int = None,  # Sparsity pattern is independent of bs and num_heads, so leave as None
    mask_num_heads: int = None,
    device: str = None,
    separate_full_blocks: bool = True
):
    assert query.ndim == 4
    assert key.ndim == 4
    assert value.ndim == 4
    assert query.shape[2] == model_config.num_heads
    assert key.shape[2] == model_config.num_key_value_heads
    assert key.shape == value.shape
    
    device = device or query.device

    q_len, kv_len = query.shape[1], key.shape[1]
    block_mask, flex_attn_func = configure_flex_attn(
        batch_size=mask_batch_size,
        num_heads=mask_num_heads,
        q_seq_len=q_len,
        kv_seq_len=kv_len,
        compile_config=compile_config,
        device=device,
        separate_full_blocks=separate_full_blocks
    )
    flex_runner = partial(run_flex_attn, enable_gqa=model_config.num_key_value_groups > 1, backwards=backwards)
    t_us = benchmark_torch_function_in_microseconds(
        flex_runner, flex_attn_func, block_mask, query, key, value
    )
    return t_us


def clone_inputs(*args, requires_grad: bool = False):
    return [t.detach().clone().requires_grad_(requires_grad) for t in args]

def run_benchmarks(
    model_config: ModelConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sdpbackend: SDPBackend = SDPBackend.FLASH_ATTENTION,
    flex_attn_compile_config: dict = None,
    backwards: bool = False,
):
    q, k, v = clone_inputs(query, key, value, requires_grad=backwards)
    t_us = benchmark_sdpa(model_config, q, k, v, backend=sdpbackend, backwards=backwards)
    flavor = "fwd-bwd" if backwards else "fwd"

    print(f"SDPA attn with {sdpbackend}, {flavor}: {t_us:.2f}us")

    q, k, v = clone_inputs(query, key, value, requires_grad=backwards)
    t_us = benchmark_flex_attn(
        model_config, q, k, v, compile_config=flex_attn_compile_config, backwards=backwards
    )
    print(f"Flex attn with {flex_attn_compile_config}, {flavor}: {t_us:.2f}us")

if __name__ == "__main__":
    from torch._logging import set_logs
    set_logs(recompiles_verbose=True)

    config = ModelConfig()
    backwards = True

    query, key, value = make_inputs(config, device=DEVICE, dtype=DTYPE)

    sdpbackend = SDPBackend.FLASH_ATTENTION
    flex_attn_compile_config = {"fullgraph": False, "dynamic": True}
    run_benchmarks(
        config,
        query,
        key,
        value,
        sdpbackend=sdpbackend,
        flex_attn_compile_config=flex_attn_compile_config,
        backwards=backwards,
    )
