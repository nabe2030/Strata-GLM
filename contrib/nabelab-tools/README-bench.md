# nabelab-tools: benchmark helpers for Strata-GLM

Small stand-alone Python 3 scripts used for the GLM measurements. No dependencies beyond the standard library unless noted. Run them with `python3 -I` or from a clean directory.

## m4_recompute.py
Recomputes throughput and TTFT from the **server's engine log**, because the counters reported by `vllm bench serve` are wrong for Strata (it under-counts generated tokens and reports a meaningless TTFT).
- Usage: `python3 m4_recompute.py ./results/run1` — the directory must hold `c1.json` (the `--save-result --save-detailed` output of a c=1, 200-prompt run), optionally `c2.json` (c=2, 50 prompts, run right after), and `engine_requests.txt`.
- Prerequisite: `engine_requests.txt` is `grep 'strata serve: request:' <engine log>`. Expected line format: `request: prompt N (reused R) in T ms, G tokens in Y ms (S tok/s), <finish reason>`. The last 200 lines (c1 only) or last 250 lines (c1+c2) are matched to the client's requests; the script checks that server prompt length minus client prompt length is constant.
- Output: `m4_recomputed.json` and a printed table (output/total tok/s over wall time, decode tok/s, server-side prompt-read time as TTFT: median/mean/p99, finish reasons).

## fn_recompute.py
Same idea for the log format of the Flash-Next engine build.
- Usage: `python3 fn_recompute.py ./results/run1` with `c1.json` and `engine_requests.txt` (last 200 lines = the 200 c=1 requests).
- Expected line format: `strata serve: prompt N tokens = R reused + X read in T ms (...), G generated in Y ms (S tok/s)[, drafts accepted a of b]`.
- Output: `fn_recomputed.json` and a printed table, including the speculative-draft acceptance rate when present.

## jamcqa_resume.py
Runs JamC-QA v1.0 (test split, 2,309 questions) against an OpenAI-compatible chat endpoint, one request at a time, 3-shot, answer = first digit 0-3 in the reply.
- Appends every question to `<outdir>/results.jsonl` and resumes from there after a restart; communication/API errors are retried with back-off and never counted as wrong answers.
- Usage: `pip install datasets httpx`, then `API_URL=http://localhost:8095/v1/chat/completions python3 jamcqa_resume.py --outdir ./results/jamcqa --progress-file ./results/jamcqa/progress.txt` (`--limit N` for a smoke test, `--extra-body` to change the chat-template kwargs; default disables thinking).
- Data: downloaded automatically from the Hugging Face hub (`sbintuitions/JamC-QA`, config `v1.0`; dev split supplies the shots). Writes `summary.json` with overall/per-category accuracy, answer-position counts and format-failure counts.
