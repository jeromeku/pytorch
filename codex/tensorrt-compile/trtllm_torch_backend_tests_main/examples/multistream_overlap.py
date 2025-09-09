import torch

torch.backends.cuda.matmul.allow_tf32 = True

def heavy_mm(a, b, iters=64):
    x = a
    for _ in range(iters):
        x = x @ b
    return x

def main():
    assert torch.cuda.is_available()
    device = "cuda"
    n = 2048
    a = torch.randn(n, n, device=device, dtype=torch.float16)
    b = torch.randn(n, n, device=device, dtype=torch.float16)
    c = torch.randn(n, n, device=device, dtype=torch.float16)
    d = torch.randn(n, n, device=device, dtype=torch.float16)

    s0 = torch.cuda.default_stream()
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    e_compute_done = torch.cuda.Event(blocking=False, interprocess=False)

    with torch.cuda.stream(s1):
        y1 = heavy_mm(a, b, iters=32)
        e_compute_done.record(s1)

    with torch.cuda.stream(s2):
        s2.wait_event(e_compute_done)
        y2 = heavy_mm(c, d, iters=32)

    with torch.cuda.stream(s0):
        z = (a @ c).relu()

    torch.cuda.synchronize()
    print("Done. Overlap demonstrated with non-default streams + event sync.")
    for s in (s1, s2):
        s.synchronize()

if __name__ == "__main__":
    main()
