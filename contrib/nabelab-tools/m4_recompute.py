#!/usr/bin/env python3
"""vllm bench serve の指標は Strata で過少計数になるため、Strata サーバーのエンジンログの `request:` 行から再計算する（引数: 結果ディレクトリ。中に c1.json、任意で c2.json、engine_requests.txt）。
c1 = ログの 200 件、c2 = 末尾の 50 件（c2.json が無ければ c1 = 末尾の 200 件のみ）。位置合わせは入力長（サーバー = クライアント + 13）で検証済み。"""
import re, json, statistics as st, sys, os
d = sys.argv[1]
R = re.compile(r"request: prompt (\d+) \(reused (\d+)\) in (\d+) ms, (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\), (\w+)")
S = []
for l in open(d + "/engine_requests.txt"):
    m = R.search(l)
    if m: S.append((int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), float(m[6]), m[7]))
def pct(v, p): v = sorted(v); return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]
out = {}
HAS_C2 = os.path.exists(d + "/c2.json")
for name, n in (("c1", 200), ("c2", 50)):
    if name == "c2" and not HAS_C2: continue
    c = json.load(open(f"{d}/{name}.json"))
    off = len(S) - 50 if name == "c2" else len(S) - (250 if HAS_C2 else 200)
    seg = S[off:off + n]
    assert len(seg) == n
    delta = {s[0] - i for s, i in zip(seg, c["input_lens"])}
    gen = sum(s[3] for s in seg); dec = sum(s[4] for s in seg) / 1000; prm = sum(s[2] for s in seg) / 1000
    dur = c["duration"]; pin = sum(s[0] for s in seg)
    r = {"件数": n, "入力長差(サーバー-クライアント)": sorted(delta), "wall秒": round(dur, 1), "サーバー入力トークン合計": pin, "サーバー生成トークン合計": gen,
         "vllm報告の生成トークン": c["total_output_tokens"], "vllm報告のoutput tok/s": round(c["output_throughput"], 2),
         "再計算 output tok/s (生成/wall)": round(gen / dur, 2), "再計算 total tok/s ((入力+生成)/wall)": round((pin + gen) / dur, 2),
         "decode tok/s (生成/decode秒合計)": round(gen / dec, 2), "要求ごとのdecode tok/s 中央値": round(st.median(s[5] for s in seg if s[3] >= 20), 2),
         "要求ごとのdecode tok/s 最小/最大(20tok以上)": [round(min(s[5] for s in seg if s[3] >= 20), 2), round(max(s[5] for s in seg if s[3] >= 20), 2)],
         "サーバー側 prompt読み時間(=TTFT相当, ms) 中央値": st.median(s[2] for s in seg), "同 p99": pct([s[2] for s in seg], 99), "同 平均": round(st.mean(s[2] for s in seg)),
         "同 最大": max(s[2] for s in seg), "prompt読み合計秒": round(prm, 1), "decode合計秒": round(dec, 1),
         "終了理由": {k: sum(1 for s in seg if s[6] == k) for k in {s[6] for s in seg}}, "reused>0 の件数": sum(1 for s in seg if s[1] > 0),
         "失敗": c["failed"], "vllmのTTFT中央値ms(無効)": round(c["median_ttft_ms"], 2),
         "vllm出力0トークンと数えた件数": sum(1 for o in c["output_lens"] if o == 0), "本文が空だった件数": sum(1 for t in c["generated_texts"] if not t)}
    if name == "c2":
        # c=2: キュー待ちを含む実測の要求あたり所要（クライアント: ttft+sum(itl)）と、c=1 と同じ要求との比較
        lat = [t + sum(i) for t, i in zip(c["ttfts"], c["itls"])]
        r["クライアント側の要求あたり所要秒 中央値"] = round(st.median(lat), 1)
        c1 = json.load(open(f"{d}/c1.json")); l1 = [t + sum(i) for t, i in zip(c1["ttfts"], c1["itls"])][:50]
        r["c=1 の同じ50件の所要秒 中央値"] = round(st.median(l1), 1)
        c1seg = S[len(S) - 250:len(S) - 250 + 50]
        r["c=1 の同じ50件: 生成/wall の相当 (生成/(prompt+decode秒))"] = round(sum(s[3] for s in c1seg) / (sum(s[2] + s[4] for s in c1seg) / 1000), 2)
        r["c=2: 生成/(prompt+decode秒合計)"] = round(gen / (prm + dec), 2)
    out[name] = r
json.dump(out, open(d + "/m4_recomputed.json", "w"), ensure_ascii=False, indent=1)
for k, v in out.items():
    print("==", k)
    for a, b in v.items(): print(f"  {a}: {b}")
