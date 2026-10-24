#!/usr/bin/env python3
"""JamC-QA v1.0 (test 2,309問) を OpenAI互換APIで1並列で回す。
公開されている JamC-QA の評価手順に沿ったもの（3-shot、番号のみ回答、先頭の 0〜3 を抽出して採点）。
変更点: (1) 問ごとに results.jsonl へ追記し、再起動で続きから再開 (2) 通信エラーは記録せず待って再試行（正誤に混ぜない）
(3) 途中経過を progress ファイルへ (4) 選択肢の位置分布とカテゴリ別を集計 (5) 認証ヘッダは任意。"""
import json, time, argparse, os, sys, httpx
from datasets import load_dataset

def build_prompt(q, choices):
    return "\n".join([f"質問: {q}"] + [f"{i}: {c}" for i, c in enumerate(choices)] + ["回答（番号のみ）:"])

def build_messages(q, choices, few_shot):
    m = [{"role": "system", "content": "あなたは日本に関する知識問題に答えるアシスタントです。選択肢の番号のみを回答してください。"}]
    for ex in few_shot:
        m.append({"role": "user", "content": build_prompt(ex["question"], ex["choices"])})
        m.append({"role": "assistant", "content": str(ex["answer"])})
    m.append({"role": "user", "content": build_prompt(q, choices)})
    return m

def extract_answer(t):
    if not t: return -1
    for ch in t.strip():
        if ch in "0123": return int(ch)
    return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8095/v1/chat/completions"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "glm-5.3-flash-strata"))
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--num-shots", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--extra-body", default='{"chat_template_kwargs":{"enable_thinking":false}}')
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--progress-file", default=None)
    ap.add_argument("--report-every", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    jl = os.path.join(a.outdir, "results.jsonl")
    extra = json.loads(a.extra_body)
    dev = list(load_dataset("sbintuitions/JamC-QA", "v1.0", split="dev"))
    few = [{"question": e["question"], "choices": [e[f"choice{j}"] for j in range(4)], "answer": e["answer_index"]} for e in dev[:a.num_shots]]
    items = list(load_dataset("sbintuitions/JamC-QA", "v1.0", split="test"))
    if a.limit: items = items[:a.limit]
    done = {}
    if os.path.exists(jl):
        for l in open(jl):
            if l.strip():
                r = json.loads(l); done[r["idx"]] = r
    print(f"全{len(items)}問、再開時点で完了 {len(done)} 問", flush=True)
    cli = httpx.Client(timeout=a.timeout)
    t_start = time.time(); t_run = 0.0; n_run = 0
    out = open(jl, "a")
    for i, it in enumerate(items):
        if i in done: continue
        choices = [it[f"choice{j}"] for j in range(4)]
        body = {"model": a.model, "messages": build_messages(it["question"], choices, few), "max_tokens": a.max_tokens, "temperature": 0.0, **extra}
        t0 = time.time(); tries = 0
        while True:
            try:
                resp = cli.post(a.api_url, json=body); data = resp.json()
                if "choices" in data: break
                err = json.dumps(data, ensure_ascii=False)[:300]
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            tries += 1
            print(f"[{i+1}] 通信/API エラー({tries}回目): {err}", flush=True)
            with open(os.path.join(a.outdir, "errors.log"), "a") as f: f.write(f"{time.strftime('%F %T')} idx={i} try={tries} {err}\n")
            if tries >= 20: print("20回連続エラー。中止（再起動で再開可）", flush=True); sys.exit(2)
            time.sleep(min(60 * tries, 300))
        msg = data["choices"][0]["message"]; content = msg.get("content") or ""; reasoning = msg.get("reasoning_content") or ""
        fin = data["choices"][0].get("finish_reason", "")
        pred = extract_answer(content); lat = time.time() - t0
        r = {"idx": i, "qid": it["qid"], "category": it["category"], "gold": it["answer_index"], "pred": pred,
             "correct": pred == it["answer_index"], "response": content, "reasoning_len": len(reasoning), "finish": fin,
             "latency": round(lat, 2), "retries": tries, "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
             "cached_tokens": (data.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
             "t": time.strftime("%F %T")}
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush(); done[i] = r
        t_run += lat; n_run += 1
        n = len(done); c = sum(1 for x in done.values() if x["correct"])
        if n % 10 == 0 or n == len(items):
            rem = (len(items) - n) * (t_run / n_run) / 3600
            line = f"{time.strftime('%F %T')} {n}/{len(items)}問 正解率{c/n*100:.2f}% 平均{t_run/n_run:.1f}秒/問 残り約{rem:.1f}時間 形式崩れ{sum(1 for x in done.values() if x['pred']<0)} 打切り{sum(1 for x in done.values() if x['finish']=='length')}"
            print(line, flush=True)
            if a.progress_file: open(a.progress_file, "w").write(line + "\n")
    rs = [done[i] for i in range(len(items))]
    n = len(rs); c = sum(r["correct"] for r in rs)
    cat = {}
    for r in rs:
        d = cat.setdefault(r["category"], [0, 0]); d[0] += 1; d[1] += r["correct"]
    pos = [sum(1 for r in rs if r["pred"] == k) for k in range(4)]
    summ = {"dataset": "sbintuitions/JamC-QA v1.0 test", "n": n, "correct": c, "accuracy": round(c / n * 100, 2),
            "parse_fail": sum(1 for r in rs if r["pred"] < 0), "finish_length": sum(1 for r in rs if r["finish"] == "length"),
            "empty_content": sum(1 for r in rs if not r["response"].strip()), "reasoning_nonempty": sum(1 for r in rs if r["reasoning_len"] > 0),
            "think_marker_in_content": sum(1 for r in rs if "<think>" in r["response"] or "</think>" in r["response"]),
            "retried_questions": sum(1 for r in rs if r["retries"]), "pred_position_counts": pos,
            "gold_position_counts": [sum(1 for r in rs if r["gold"] == k) for k in range(4)],
            "by_category": {k: {"n": v[0], "correct": v[1], "accuracy": round(v[1] / v[0] * 100, 2)} for k, v in cat.items()},
            "mean_latency_s": round(sum(r["latency"] for r in rs) / n, 2), "extra_body": extra, "max_tokens": a.max_tokens,
            "num_shots": a.num_shots, "model": a.model}
    json.dump(summ, open(os.path.join(a.outdir, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "by_category"}, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
