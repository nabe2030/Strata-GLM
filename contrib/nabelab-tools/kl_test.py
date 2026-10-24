#!/usr/bin/env python3
"""kl_test.py PROMPT_FILE [--url URL] [--model NAME] [--n 48] [--short] [--out order.json]

Teacher-forced quality comparison driver. Start the server with STRATA_SERVE_FOLLOW=<ids file> and
STRATA_SERVE_LOGITS=<logits.bin> (hooks in the linux-2gpu branch); every request then appends its
logit rows to logits.bin. This script sends, in order:
  A:P1  B:P2 (continues P1; prompt snapshot restore)  C:unrelated short request (interrupt)
  D:P2 from scratch  E:P2 again (snapshot)  C2  F:P2 from scratch again
and writes which rows of logits.bin belong to which request (order.json). Feed order.json and
logits.bin to kl_split.py. --short runs A,B,C,D only (long prompts: each cold read is minutes).
Pure standard library; the prompt is read from PROMPT_FILE (plain text)."""
import argparse, json, os, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("prompt_file"); ap.add_argument("--url", default=os.environ.get("STRATA_URL", "http://127.0.0.1:8095/v1"))
ap.add_argument("--model", default="glm-5.3-flash-strata"); ap.add_argument("--n", type=int, default=48)
ap.add_argument("--short", action="store_true"); ap.add_argument("--out", default="order.json")
ap.add_argument("--logits", default=os.environ.get("STRATA_SERVE_LOGITS", "logits.bin"), help="logits file the server appends to (to count existing rows)")
a = ap.parse_args(); V = 154880
rows = os.path.getsize(a.logits) // 4 // V if os.path.exists(a.logits) else 0
order = []
def chat(msgs):
    body = json.dumps({"model": a.model, "messages": msgs, "max_tokens": a.n, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(a.url + "/chat/completions", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))
def go(name, msgs):
    global rows
    r = chat(msgs); u = r.get("usage", {}); n = u.get("completion_tokens", 0)
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(name, "prompt", u.get("prompt_tokens"), "cached", cached, "gen", n, flush=True)
    order.append(dict(name=name, start=rows, n=n, prompt=u.get("prompt_tokens"), cached=cached)); rows += n
u1 = [{"role": "user", "content": open(a.prompt_file).read()}]
p2 = u1 + [{"role": "assistant", "content": "Understood."}, {"role": "user", "content": "Now list the three most important points as bullet points."}]
side = [{"role": "user", "content": "Write a short title (max 6 words) for a chat about reading documents."}]
go("A_p1", u1); go("B_p2_resume", p2); go("C_side", side); go("D_p2_full", p2)
if not a.short:
    go("E_p2_repeat", p2); go("C2_side", side); go("F_p2_full2", p2)
json.dump(order, open(a.out, "w"), indent=1)
