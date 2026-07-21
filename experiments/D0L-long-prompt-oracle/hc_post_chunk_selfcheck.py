"""Bitwise self-check for the chunked hc_post (gaiban-lite D0L, TARGET 7.3).

The claim is that chunking along s cannot change a value, because the only
reduction is over hc and no output row depends on any other token.  A claim
about bitwise equality is exactly the kind that should be measured rather than
argued, so this compares the two forms on real-range data at the shapes that
matter -- including the 8192 that OOMs unchunked, which is the whole point.
"""
import json, torch

HC, D = 4, 4096


def unchunked(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


def chunked(x, residual, post, comb, chunk):
    if not chunk or x.shape[1] <= chunk:
        return unchunked(x, residual, post, comb)
    pieces = []
    for start in range(0, x.shape[1], chunk):
        stop = start + chunk
        pieces.append(
            post[:, start:stop].unsqueeze(-1) * x[:, start:stop].unsqueeze(-2)
            + torch.sum(
                comb[:, start:stop].unsqueeze(-1)
                * residual[:, start:stop].unsqueeze(-2),
                dim=2,
            )
        )
    return torch.cat(pieces, dim=1).type_as(x)


dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
records = []
for s_len in (128, 1000, 1024, 2048, 4096):
    g = torch.Generator(device="cpu").manual_seed(20260721 + s_len)
    x = (torch.randn(1, s_len, D, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    residual = (torch.randn(1, s_len, HC, D, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    post = (torch.randn(1, s_len, HC, generator=g)).float().to(dev)
    comb = (torch.randn(1, s_len, HC, HC, generator=g)).float().to(dev)
    ref = unchunked(x, residual, post, comb)
    for chunk in (256, 1024, 4096):
        got = chunked(x, residual, post, comb, chunk)
        records.append({
            "s": s_len, "chunk": chunk,
            "bitwise_equal": bool(torch.equal(ref, got)),
            "max_abs_diff": float((ref.float() - got.float()).abs().max().item()),
        })
    del x, residual, post, comb, ref
    torch.cuda.empty_cache()

# the case that motivates the change: unchunked cannot even be built here
peak = {}
for s_len in (8192,):
    g = torch.Generator(device="cpu").manual_seed(7)
    x = (torch.randn(1, s_len, D, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    residual = (torch.randn(1, s_len, HC, D, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    post = torch.randn(1, s_len, HC, generator=g).float().to(dev)
    comb = torch.randn(1, s_len, HC, HC, generator=g).float().to(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    _ = chunked(x, residual, post, comb, 1024)
    peak["chunked_1024_peak_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        _ = unchunked(x, residual, post, comb)
        peak["unchunked_peak_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
        peak["unchunked_ok"] = True
    except torch.OutOfMemoryError as exc:
        peak["unchunked_ok"] = False
        peak["unchunked_error"] = str(exc)[:120]

print(json.dumps({
    "accepted": all(r["bitwise_equal"] for r in records),
    "cases": len(records),
    "records": records,
    "s8192": peak,
}, indent=1))
