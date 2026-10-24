#!/usr/bin/env python3
"""mtp_model.py — back-of-envelope generation speed with MTP of depth d.
Window time W(T) = G + C*c(T), c(T) = 1 + s*(T-1)  (s = fraction of extra tokens in a window that pull new experts).
G = GPU + sync (~40 ms), C = CPU experts (~170 ms): measured from the server log "time front/back" lines on the reference box.
s: 0.90 = free text (c(2)=1.90), 0.67 = copy-heavy windows (regressed from suffix windows with union_stats.py).
Acceptance: first draft p1, later ones conditional q; tokens per round = 1 + p1 + p1*q + ... ; numbers below are inputs, replace with yours."""
import sys
G, C = 40.0, 170.0; W1 = G + C
def run(label, p1, q, s, draft_ms, extra_ms=0.0):
    print(f"\n[{label}] p1={p1} q={q} s={s} draft {draft_ms}ms/round extra {extra_ms}ms")
    for d in (1, 2, 3, 5):
        T = d + 1; acc = p1; en = 1.0 + acc; a = acc
        for _ in range(d - 1): a *= q; en += a
        W = G + C * (1 + s * (T - 1)) + draft_ms * d / 1 * (1 if d == 1 else 0.8) + extra_ms
        print(f"  depth {d}: {en:.2f} tokens per round / {W:.0f} ms  → {en/W*1000:.2f} tok/s  (baseline {1000/W1:.2f})  speedup {en*W1/W:.2f}")
run("A: draft without routed experts, free text", 0.635, 0.58, 0.90, 13)
run("A: same, copy-heavy text", 0.635, 0.58, 0.67, 13)
run("B(CPU experts): acceptance 0.77  free text", 0.771, 0.62, 0.90, 16, 5)
run("B(CPU experts): acceptance 0.77  copy-heavy text", 0.771, 0.62, 0.67, 16, 5)
run("B optimistic: acceptance 0.93 (upstream bench, English) copy-heavy text", 0.93, 0.93, 0.67, 16, 5)
run("Japanese/code assumption: acceptance 0.56", 0.56, 0.5, 0.90, 16, 5)
# AcceptMoE-style (--spec-tail-topk 4): rows after the first use only the top-4 experts -> extra-row cost s ~ 0.45 (lossy, acceptance -2pt)
run("B(CPU experts)+tail-topk4: acceptance 0.75 free text (lossy)", 0.75, 0.60, 0.45, 16, 5)
run("A(draft without experts)+tail-topk4: acceptance 0.615 (lossy)", 0.615, 0.56, 0.45, 13)
