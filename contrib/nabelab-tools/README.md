# nabelab-tools — measurement helpers used for the two-GPU write-up

Not part of `patches/linux-2gpu`; independent of the engine code. Comments are English; `pcie_bench` prints Japanese labels.

## Teacher-forced quality comparison (needs the `STRATA_SERVE_FOLLOW` / `STRATA_SERVE_LOGITS` hooks from this branch)
1. Greedy-decode a reference once (e.g. single GPU, sequential path) and save the token ids; start the server under test with
   `STRATA_SERVE_FOLLOW=<ref ids file> STRATA_SERVE_LOGITS=logits.bin` (one float32 row of 154,880 logits per forced token).
2. `kl_test.py prompt.txt --out order.json` sends A (prompt) / B (continuation, snapshot restore) / C (interrupt) / D (from scratch) / E, F and records which rows belong to which request.
3. `kl_split.py order.json logits.bin outdir` splits the file and prints the KL between request pairs (snapshot restore vs from scratch, run to run).
4. `klrow.py ref.bin x.bin ...` compares any two logits files: mean KL, median, top-1 agreement, positions with KL > 0.1, bit-identical or not.

## Hardware micro-benchmarks
- `pcie_bench.cu` — pinned host<->GPU copies, mapped (zero-copy) access, GPU<->GPU, concurrent transfers, latency. `nvcc -O2 -o pcie_bench pcie_bench.cu`; run it with the GPUs awake.
- `bw.c` — multi-thread DRAM read bandwidth (AVX2). `mb.c` — IQ3_S/IQ4_XS x Q8_K dot-product kernel throughput per thread (links against the ggml quant code).

## MTP estimate
- `union_stats.py LOG...` — per verify-window distinct (layer, expert) pairs from `--serve` engine logs. `mtp_model.py` — speed model G + C*c(T); inputs are in the file.

## Benchmark helpers (see README-bench.md)
- `m4_recompute.py`, `fn_recompute.py` — recompute throughput/TTFT from the server's engine log (the counters of `vllm bench serve` are wrong for Strata). `jamcqa_resume.py` — resumable JamC-QA runner for an OpenAI-compatible endpoint.

## Operations
- `memwatch.sh [PIDFILE] [MAX_GPU_C] [MIN_AVAIL_MIB]` stops the server at 85 C or when MemAvailable stays under 2500 MiB. `start.sh` / `stop.sh` are skeletons (set `STRATA_DIR`, `RUN_DIR`, `VENV_PYTHON`).
- `setup-cuda-root.sh VENV CUDA_ROOT` builds a toolkit-shaped symlink tree from the pip `nvidia-*` wheels.
