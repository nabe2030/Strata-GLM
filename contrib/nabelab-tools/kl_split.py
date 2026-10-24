#!/usr/bin/env python3
"""kl_split.py order.json logits.bin OUTDIR — split the logits file into one .bin per request and print
KL divergences between request pairs (reference = D_p2_full, the from-scratch run)."""
import sys, json, os, numpy as np
V = 154880
o = json.load(open(sys.argv[1])); A = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, V)
os.makedirs(sys.argv[3], exist_ok=True)
d = {x["name"]: A[x["start"]:x["start"] + x["n"]] for x in o}
for k, v in d.items(): v.tofile(os.path.join(sys.argv[3], k + ".bin"))
def lsm(x): x = x.astype(np.float64); m = x.max(1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))
def cmp(ref, x):
    a, b = d[ref], d[x]; la, lb = lsm(a), lsm(b); kl = (np.exp(la) * (la - lb)).sum(1)
    return f"{x:14s} vs {ref:12s} mean KL {kl.mean():.5f} max KL {kl.max():.5f} top-1 {100*(a.argmax(1)==b.argmax(1)).mean():5.1f}% bit-identical {'yes' if np.array_equal(a,b) else 'no'}"
for ref, x in [("D_p2_full", "F_p2_full2"), ("D_p2_full", "B_p2_resume"), ("D_p2_full", "E_p2_repeat"), ("F_p2_full2", "B_p2_resume"), ("F_p2_full2", "E_p2_repeat"), ("C_side", "C2_side")]:
    if ref in d and x in d: print(cmp(ref, x))
