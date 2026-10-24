#!/usr/bin/env python3
"""union_stats.py LOG... — from engine logs of `--serve` runs, collect per verify-window: tokens, windows,
CPU-expert rows and distinct (layer, expert) pairs (the union). rows = pairs summed over all tokens in the window;
distinct = after removing duplicates per layer. Output CSV: log,prompt,tokens,windows,tok_per_win,rows,distinct,ratio,mb"""
import re, sys
out = ["log,prompt,tokens,windows,tok_per_win,rows,distinct,ratio,mb"]
for fn in sys.argv[1:]:
    cur = None
    for l in open(fn, errors="replace"):
        m = re.match(r"strata serve: request: prompt (\d+) \(reused (\d+)\) in [\d.]+ ms, (\d+) tokens in", l)
        if m: cur = dict(prompt=int(m[1]), tokens=int(m[3])); continue
        if cur is None: continue
        m = re.match(r"strata serve: time front layers: (\d+) windows", l)
        if m: cur["w"] = int(m[1]); continue
        m = re.match(r"strata serve: CPU experts per window: ([\d.]+) distinct \(([\d.]+) rows\), ([\d.]+) MB", l)
        if m and "w" in cur:
            d, r, mb = float(m[1]), float(m[2]), float(m[3]); w = cur["w"]
            out.append(f"{fn.split('/')[-1]},{cur['prompt']},{cur['tokens']},{w},{cur['tokens']/w:.3f},{r:.1f},{d:.1f},{d/r:.4f},{mb:.0f}")
            cur = None
print("\n".join(out))
