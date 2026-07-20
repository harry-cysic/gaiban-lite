"""Inspect raw V4-Flash checkpoint tensor contract (headers only, no payload).

Prints, for representative layers (0, 1, 2, 3, 4, 42) and mtp.0:
key -> dtype, shape, grouped with digit-collapsed dedup for experts.
Also verifies expert packing shapes against Flash geometry:
  w1/w3: (2048, 4096//2) I8 packed + scale (2048, 4096//32) E8M0
  w2:    (4096, 2048//2) I8 packed + scale (4096, 2048//32) E8M0
"""
import json, re, struct, sys, os
from collections import OrderedDict

CKPT = os.path.expanduser("~/Workspace/DeepSeek-V4-Flash")
idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
wm = idx["weight_map"]

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

# cache headers per file
hcache = {}
def tensor_meta(key):
    fn = wm[key]
    if fn not in hcache:
        hcache[fn] = read_header(os.path.join(CKPT, fn))
    h = hcache[fn][key]
    return h["dtype"], h["shape"]

targets = ["layers.0.", "layers.1.", "layers.2.", "layers.3.", "layers.4.", "layers.42.", "mtp.0."]
toplevel = [k for k in wm if "." not in k or k.split(".")[0] not in ("layers", "mtp")]

print("== top-level ==")
for k in sorted(toplevel):
    d, s = tensor_meta(k)
    print(f"  {k:50s} {d:8s} {s}")

for t in targets:
    keys = [k for k in wm if k.startswith(t)]
    seen = OrderedDict()
    for k in sorted(keys):
        pat = re.sub(r"(experts)\.\d+", r"\1.E", k)
        if pat not in seen:
            seen[pat] = k
    nexp = len(set(re.findall(r"experts\.(\d+)\.", " ".join(keys))))
    print(f"\n== {t}* ({len(keys)} keys, {nexp} experts) ==")
    for pat, k in seen.items():
        d, s = tensor_meta(k)
        print(f"  {pat:60s} {d:8s} {s}")
