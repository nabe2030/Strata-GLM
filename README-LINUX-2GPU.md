# Strata-GLM on Linux with two consumer GPUs (branch `linux-2gpu`)

This branch is [isichan-ai/Strata-GLM](https://github.com/isichan-ai/Strata-GLM) at `6feb85f` plus 10 small commits
(`patches/linux-2gpu/0001..0010-*.patch`) that let GLM-5.3-Flash run on **Linux** with **two 16 GB GPUs** in one process.
The upstream license (MIT) and the original author's credit are unchanged. Write-up (Japanese): https://qiita.com/nabe2030/items/404240ab6b9171565c9d

## What changed

| # | Patches | What |
|---|---|---|
| 1 | 0001 | **Linux build**: replaces three Windows-only calls (`_fseeki64` etc.). |
| 2 | 0002, 0003, 0005 | **2-GPU layer split** in one process: `--layer-split K --split-device D` runs layers `[0,K)` on CUDA0 and `[K,45)` plus the head on CUDA1. No P2P needed (GeForce): hand-off goes through pinned, mapped host memory (about 82 KB per token). Chunked prefill walks a layer range per call (a 1300-token chunk prefill went from 225 s to 78 s). `--trim-stage-weights` frees the weights a stage does not use. |
| 3 | 0004, 0008 | **cuBLAS binding**: a per-device cuBLAS handle and shared-memory opt-ins, and the handle is always bound to the call's compute stream (a small chunk could otherwise run on a handle that is not bound to the compute stream; see upstream issue #2). |
| 4 | 0009 | **Prompt snapshot on 2 GPUs**: the recurrent-state snapshot is kept on a local two-GPU split, so a continued conversation re-reads only the new tokens (32K-token chat, next turn: about 2-3 s instead of about 160 s). |
| 5 | 0010 | **`</think>` close for thinking-off**: when `enable_thinking` is false and the chat template ends the generation prompt in `<think>` (GLM's does), the server appends `</think>` so the reasoning does not leak into `content`. |
| - | 0006, 0007 | Investigation hooks (**environment-gated, off by default**, no behaviour change without them): `STRATA_GLM5_LDUMP/_POS/_LAYER`, `STRATA_GLM5_PF_ROUTER_CHECK`, `STRATA_GLM5_PF_BLAS_STREAM`, `STRATA_GLM5_PF_WS_SEP`, `STRATA_GLM5_PF_EXIT`, `STRATA_GLM5_PF_POISON`. Also `STRATA_SERVE_FOLLOW` / `STRATA_SERVE_LOGITS` (teacher forcing, added in 0004) for quality comparisons. |

`contrib/nabelab-tools/` (last commit, not part of `patches/`) has the measurement scripts used for the write-up.

## Build (Ubuntu 22.04, CUDA 13.4 from pip, no sudo)

The CUDA toolkit is assembled from pip wheels inside a venv; nothing is installed system-wide.

```bash
python3 -m venv ~/strata-venv && . ~/strata-venv/bin/activate
pip install -r requirements.txt                      # cmake, ninja, ...
pip install nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cublas nvidia-cuda-cccl   # 13.x (tested 13.4.92)
contrib/nabelab-tools/setup-cuda-root.sh ~/strata-venv ~/cuda-root                  # symlinks in a toolkit layout
export PATH=$HOME/strata-venv/bin:$HOME/cuda-root/bin:$PATH CUDA_PATH=$HOME/cuda-root CUDA_HOME=$HOME/cuda-root
cmake -G Ninja -S . -B build-sm89 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc -DCUDAToolkit_ROOT=$CUDA_PATH
cmake --build build-sm89 --target strata -j 8        # about 4 minutes; ~230 warnings, no errors
export LD_LIBRARY_PATH=$HOME/cuda-root/lib64
```
`sm_89` is the RTX 4060 Ti; change `CMAKE_CUDA_ARCHITECTURES` for other cards. Make the pack and a config with
`tools/glm5_ref/make_glm5_config.py` (see upstream README), then start from `strata-glm5.linux-2gpu.example.json`.

## Running on two GPUs

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # keep CUDA0/CUDA1 in PCI order
python serve/server.py --engine strata --config strata-glm5.json     # config from the example file
```
Engine arguments that matter (all in the example file): `--layer-split 12 --split-device 1 --trim-stage-weights`.
- `--resident-budget-gib 95` keeps 95 GiB of experts in page-locked RAM (128 GB box; leave ~30 GB); `--vram-reserve-mib 3400` is scratch for the chunked prefill; `--max-context 262144`; chunk 4096 (`STRATA_GLM5_PREFILL=4096`).
- `STRATA_GLM5_PF_MIN=256`: a continuation with fewer than 256 new tokens uses the sequential path (break-even estimated at about 190 new tokens; 256 was the value we measured with), longer ones the chunked path.
- `--pool-workers 10` on a 6-core/12-thread CPU. Using all 12 threads was 33 % **slower**.
- **Choosing K**: put fewer layers on the GPU behind the slower PCIe link. On our first board (GPU0 behind the chipset at Gen3 x4, GPU1 at Gen4 x8) K=12 was among the best and K=30 was 13 % slower than K=12 on a 4K prompt (91.6 s vs 81.2 s). With both GPUs at Gen4 x8 the 4,147-token prompt took 64-65 s for K=8, 12, 17 and 23 (within 1.5 %), so **with symmetric links K does not matter**; K=12 was kept.
- Link speed of a 4060 Ti drops to Gen1 at idle; measure under load.

## Results (RTX 4060 Ti 16 GB x 2, Ryzen 5 5600, DDR4 128 GB, both GPUs Gen4 x8, IQ4_XS, thinking off)

- Generation: about 5.5 tok/s on short prompts (5.1-6.1 over five topics), 3.4-4.2 tok/s after a 4K-32K prompt. CPU expert compute is the limit, not the GPUs or PCIe: the second board (symmetric links) did not change generation speed.
- Cold time to first token: 86 tokens 10-13 s, 560 tokens 31-35 s, 4.4K tokens 27-43 s (noisy), 32K tokens 135-147 s. Same-conversation continuation: 2-3 s at 4.4K and 32K; an identical repeat 0.2 s.
- Quality: teacher-forced KL against a single-GPU run (sequential path) is 0.01-0.03 mean (K=12: 0.010 on an 86-token prompt, 0.028 on 4,147 tokens), i.e. the same order as run-to-run differences between GPU/CPU arithmetic paths. Two runs of the same two-GPU configuration are bit-identical.
- VRAM: GPU1 has only about 0.5 GB free at 32K with these settings.

## Known limitations

- With two GPUs the prompt snapshot works for continuing conversations but is lost when an unrelated short request interrupts (that turn re-reads the whole prompt). A fix (keeping several snapshots) is not implemented.
- `--layer-split 5` (very few layers on CUDA0) does not start; K=8 and up work.
- GPU1's VRAM headroom is small; larger `--max-context` or chunk sizes can run out of memory.
- MTP is not used by `serve` in this branch.
- Tested only on Ubuntu 22.04, driver 615, CUDA 13.4, sm_89, two identical GPUs.
