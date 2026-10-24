# Strata-GLM を Linux・民生 GPU 2 枚で動かす（ブランチ `linux-2gpu`）

このブランチは [isichan-ai/Strata-GLM](https://github.com/isichan-ai/Strata-GLM) の `6feb85f` の上に、小さなコミット 10 本
（`patches/linux-2gpu/0001..0010-*.patch`）を載せたものです。GLM-5.3-Flash を **Linux** で **16GB の GPU 2 枚** を 1 プロセスで動かせるようにします。
ライセンス（MIT）と作者のクレジットは元のままです。解説記事: https://qiita.com/nabe2030/items/404240ab6b9171565c9d

## 変更点

| # | パッチ | 内容 |
|---|---|---|
| 1 | 0001 | **Linux ビルド**: Windows 専用の呼び出し 3 か所（`_fseeki64` など）の置き換え。 |
| 2 | 0002, 0003, 0005 | **2 枚の層分割**（1 プロセス）: `--layer-split K --split-device D` で層 `[0,K)` を CUDA0、`[K,45)` とヘッドを CUDA1 に置く。GeForce は P2P 不可なので受け渡しはピン留めしたホストメモリ経由（1 トークン約 82KB）。チャンク prefill は層の範囲ごとに処理（1,300 トークンで 225 秒→78 秒）。`--trim-stage-weights` で使わない重みを解放。 |
| 3 | 0004, 0008 | **cuBLAS の束縛**: デバイスごとの cuBLAS ハンドルと共有メモリ設定。ハンドルを常に呼び出しの計算ストリームに束縛（小さいチャンクで束縛されないことがあった。上流 issue #2）。 |
| 4 | 0009 | **2 枚での prompt snapshot**: 直前プロンプトの再帰状態を 2 枚構成でも保持。会話の続きは新しい分だけ読む（32K の会話で次のターン 約 2〜3 秒、以前は約 160 秒）。 |
| 5 | 0010 | **thinking 無効時の `</think>`**: `enable_thinking` が false でテンプレートが `<think>` で終わる（GLM はそう）とき `</think>` を足し、推論が `content` に漏れないようにする。 |
| - | 0006, 0007 | 調査用フック（**環境変数で有効化、既定は無効**。付けなければ挙動は変わらない）: `STRATA_GLM5_LDUMP/_POS/_LAYER`、`STRATA_GLM5_PF_ROUTER_CHECK`、`STRATA_GLM5_PF_BLAS_STREAM`、`STRATA_GLM5_PF_WS_SEP`、`STRATA_GLM5_PF_EXIT`、`STRATA_GLM5_PF_POISON`。教師強制（`STRATA_SERVE_FOLLOW` / `STRATA_SERVE_LOGITS`、0004）も品質比較用。 |

`contrib/nabelab-tools/`（最後のコミット、`patches/` には含めない）に、記事で使った測定スクリプトを置いています。

## ビルド（Ubuntu 22.04、pip の CUDA 13.4、sudo 不要）

CUDA ツールキットは venv 内の pip ホイールから組みます。システムには何も入れません。

```bash
python3 -m venv ~/strata-venv && . ~/strata-venv/bin/activate
pip install -r requirements.txt
pip install nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cublas nvidia-cuda-cccl   # 13.x（13.4.92 で確認）
contrib/nabelab-tools/setup-cuda-root.sh ~/strata-venv ~/cuda-root                  # ツールキット形のシンボリックリンク
export PATH=$HOME/strata-venv/bin:$HOME/cuda-root/bin:$PATH CUDA_PATH=$HOME/cuda-root CUDA_HOME=$HOME/cuda-root
cmake -G Ninja -S . -B build-sm89 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc -DCUDAToolkit_ROOT=$CUDA_PATH
cmake --build build-sm89 --target strata -j 8        # 約 4 分、警告 約 230 件、エラーなし
export LD_LIBRARY_PATH=$HOME/cuda-root/lib64
```
`sm_89` は RTX 4060 Ti。別のカードなら `CMAKE_CUDA_ARCHITECTURES` を変える。pack と設定は `tools/glm5_ref/make_glm5_config.py`（上流 README）で作り、
`strata-glm5.linux-2gpu.example.json` を元に起動します。

## 2 枚での起動

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # CUDA0/CUDA1 を PCI 順に固定
python serve/server.py --engine strata --config strata-glm5.json
```
要点（設定例に全部入っています）: `--layer-split 12 --split-device 1 --trim-stage-weights`。
- `--resident-budget-gib 95`: エキスパート 95GiB を page-locked RAM に常駐（128GB 機。約 30GB は残す）。`--vram-reserve-mib 3400`: チャンク prefill の作業領域。`--max-context 262144`。チャンク 4096（`STRATA_GLM5_PREFILL=4096`）。
- `STRATA_GLM5_PF_MIN=256`: 新しい分が 256 トークン未満の続きは逐次処理、それ以上は一括処理（損益分岐は約 190 トークンと見積もり、256 で測定）。
- `--pool-workers 10`（6 コア 12 スレッドの CPU）。12 本にすると 33% **遅くなった**。
- **K の決め方**: 遅い PCIe 側の GPU に置く層を少なくする。最初の板（GPU0 がチップセット経由 Gen3 x4、GPU1 が Gen4 x8）では K=12 が良好で、K=30 は K=12 より 13% 遅かった（4K で 91.6 秒 対 81.2 秒）。2 枚とも Gen4 x8 では 4,147 トークンの入力が K=8・12・17・23 で 64〜65 秒（差 1.5% 以内）で、**リンクが対称なら K は効かない**。K=12 のままにした。
- 4060 Ti のリンクはアイドルで Gen1 に落ちる。測るなら負荷中に。

## 測定結果（RTX 4060 Ti 16GB×2、Ryzen 5 5600、DDR4 128GB、両方 Gen4 x8、IQ4_XS、thinking 無効）

- 生成: 短い入力で約 5.5 tok/s（5 話題で 5.1〜6.1）、4K〜32K の入力の後は 3.4〜4.2 tok/s。律速は CPU のエキスパート計算で、GPU や PCIe ではない（対称リンクの板に替えても生成速度は変わらなかった）。
- 初回の TTFT: 86 トークン 10〜13 秒、560 トークン 31〜35 秒、4.4K 27〜43 秒（ばらつき大）、32K 135〜147 秒。同じ会話の続きは 4.4K・32K とも 2〜3 秒、同一入力の再送は 0.2 秒。
- 品質: 1 枚（逐次処理）に対する教師強制 KL は平均 0.01〜0.03（K=12: 86 トークンで 0.010、4,147 トークンで 0.028）。GPU/CPU の算術経路の違いと同じ桁。同じ 2 枚構成の 2 回の実行はビット一致。
- VRAM: この設定の 32K では GPU1 の空きが約 0.5GB。

## 既知の制限

- 2 枚での snapshot は会話の続きでは効くが、無関係な短い要求の割り込みで消える（次のターンは全体を読み直す）。複数 snapshot を持つ修正は未実装。
- `--layer-split 5`（CUDA0 の層が極端に少ない）は起動しない。K=8 以上は動く。
- GPU1 の VRAM の余裕が小さく、`--max-context` やチャンクを大きくすると不足しうる。
- このブランチの `serve` では MTP は使っていない。
- 確認した環境: Ubuntu 22.04、ドライバ 615、CUDA 13.4、sm_89、同型 GPU 2 枚のみ。
