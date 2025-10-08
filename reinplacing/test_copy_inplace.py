import torch

t = torch.randn(128, device="cuda")
t2 = torch.randn_like(t)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
) as prof:
    torch.ops.aten.copy_.default(t, t)


print(prof.key_averages().table(header="Same tensor copy"))

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
) as prof:
    torch.ops.aten.copy_.default(t, t2)


print(prof.key_averages().table(header="Different tensor copy"))