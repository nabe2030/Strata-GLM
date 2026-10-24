#!/usr/bin/env python3
# usage: klrow.py REF.bin X.bin ...  (float32 logits rows, vocab 154880)
# one line per file: mean KL(ref||x) / median / top-1 agreement / positions with KL>0.1 / bit-identical
import sys, numpy as np
V=154880
def load(fn): return np.fromfile(fn,dtype=np.float32).reshape(-1,V).astype(np.float64)
def lsm(x): m=x.max(1,keepdims=True); return x-m-np.log(np.exp(x-m).sum(1,keepdims=True))
A=load(sys.argv[1]); la=lsm(A)
for fn in sys.argv[2:]:
    try: B=load(fn)
    except Exception as e: print(fn,"unreadable"); continue
    n=min(len(A),len(B))
    if n==0: print(fn.split('/')[-1],"empty"); continue
    lb=lsm(B[:n]); kl=(np.exp(la[:n])*(la[:n]-lb)).sum(1); t1=(A[:n].argmax(1)==B[:n].argmax(1))
    print(f"{fn.split('/')[-1][:-4]:10s} mean KL {kl.mean():7.4f}  median {np.median(kl):7.4f}  top-1 {t1.mean()*100:5.1f}%  KL>0.1: {int((kl>0.1).sum()):2d}/{n}  bit-identical {'yes' if np.array_equal(A[:n],B[:n]) else 'no'}")
