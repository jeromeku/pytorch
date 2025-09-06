import os
import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    noop_mask,
)


def score_mod(score, b, h, m, n):
    # Simple, differentiable mod: add a small bias depending on (m, n)
    # Keeps data-dependent indexing obvious in the trace
    return score + (m.to(score.dtype) * 0.0 + n.to(score.dtype) * 0.0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Shapes kept small for demo; head dims >= 16 for Triton path
    B = 1
    H = 2
    M = 32
    N = 32
    D = 32

    torch.manual_seed(0)
    q = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

    # Build a dense/no-op block mask; you can swap in custom masks later
    block_mask = create_block_mask(noop_mask, B, 1, M, N, device=device)

    compiled_flex = torch.compile(
        flex_attention, backend="inductor", fullgraph=True, dynamic=False
    )

    # Optional: enable rich logs/dumps
    # os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
    # os.environ.setdefault("TORCH_LOGS", "+dynamo,inductor,aot")

    out = compiled_flex(q, k, v, score_mod=score_mod, block_mask=block_mask)
    print("out shape:", out.shape if isinstance(out, torch.Tensor) else [t.shape for t in out])

    # Backward to exercise flex_attention_backward HOP
    if isinstance(out, torch.Tensor):
        loss = out.sum()
        loss.backward()
        print("grad q nan?", torch.isnan(q.grad).any().item())


if __name__ == "__main__":
    main()

