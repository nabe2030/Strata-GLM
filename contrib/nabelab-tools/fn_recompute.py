#!/usr/bin/env python3
"""Flash-Next 版の再計算（m4_recompute.py と同じ考え方）。引数: 結果ディレクトリ。engine_requests.txt の末尾 200 行 = c1 200 件。
書式: strata serve: prompt N tokens = R reused + X read in T ms (..), G generated in Y ms (S tok/s)[, drafts accepted a of b]"""
import re, json, statistics as st, sys
d = sys.argv[1]
R = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)(?:, drafts accepted (\d+) of (\d+))?")
S = [tuple(float(x) if x and '.' in x else (int(x) if x else 0) for x in m.groups()) for m in map(R.search, open(d + "/engine_requests.txt")) if m]
def pct(v, p): v = sorted(v); return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]
c = json.load(open(d + "/c1.json")); seg = S[len(S) - 200:]
delta = sorted({int(s[0]) - i for s, i in zip(seg, c["input_lens"])})
pin = sum(s[0] for s in seg); gen = sum(s[4] for s in seg); dur = c["duration"]
dec = sum(s[5] for s in seg) / 1000; prm = sum(s[3] for s in seg) / 1000
ok = [s for s in seg if s[4] >= 20]; da = sum(s[7] for s in seg); db = sum(s[8] for s in seg)
r = {"件数": len(seg), "入力長差(サーバー-クライアント)の種類": delta, "wall秒": round(dur, 1), "サーバー入力トークン合計": pin, "サーバー生成トークン合計": gen,
     "vllm報告の生成トークン": c["total_output_tokens"], "vllm報告のoutput tok/s": round(c["output_throughput"], 2),
     "再計算 output tok/s (生成/wall)": round(gen / dur, 2), "再計算 total tok/s ((入力+生成)/wall)": round((pin + gen) / dur, 2),
     "decode tok/s (生成/decode秒合計)": round(gen / dec, 2), "要求ごとのdecode tok/s 中央値": round(st.median(s[6] for s in ok), 2),
     "要求ごとのdecode tok/s 最小/最大(20tok以上)": [round(min(s[6] for s in ok), 2), round(max(s[6] for s in ok), 2)],
     "サーバー側 prompt読み時間(=TTFT相当, ms) 中央値": st.median(s[3] for s in seg), "同 p99": pct([s[3] for s in seg], 99), "同 平均": round(st.mean(s[3] for s in seg)),
     "同 最大": max(s[3] for s in seg), "prompt読み合計秒": round(prm, 1), "decode合計秒": round(dec, 1),
     "reused>0 の件数": sum(1 for s in seg if s[1] > 0), "drafts受理(合計)": [int(da), int(db)], "drafts受理率": round(da / db, 3) if db else None,
     "失敗": c["failed"], "vllmのTTFT中央値ms(無効)": round(c["median_ttft_ms"], 2),
     "vllm出力0トークンと数えた件数": sum(1 for o in c["output_lens"] if o == 0), "本文が空だった件数": sum(1 for t in c["generated_texts"] if not t)}
json.dump(r, open(d + "/fn_recomputed.json", "w"), ensure_ascii=False, indent=1)
for k, v in r.items(): print(k, ":", v)
