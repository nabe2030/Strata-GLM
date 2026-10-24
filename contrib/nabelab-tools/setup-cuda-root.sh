#!/bin/bash
# setup-cuda-root.sh VENV CUDA_ROOT — build a CUDA-toolkit-shaped directory of symlinks from the pip nvidia-* wheels in VENV.
# Needed because the wheels have no lib64/ and no unversioned .so files. libcuda.so points at the system driver.
set -e
VENV=${1:?usage: setup-cuda-root.sh VENV CUDA_ROOT}; R=${2:?}
N=$(echo $VENV/lib/python3*/site-packages/nvidia/cu13); [ -d "$N" ] || { echo "no nvidia/cu13 under $VENV"; exit 1; }
mkdir -p $R/lib64; cd $R
for d in bin include nvvm; do ln -sfn $N/$d $d; done
ln -sfn lib64 lib
for f in $N/lib/*.so.* $N/lib/*.a; do b=$(basename $f); ln -sfn $f lib64/$b; done
for f in lib64/*.so.*; do b=$(basename $f); base=${b%%.so.*}; [ -e lib64/$base.so ] || ln -sfn $b lib64/$base.so; done
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 lib64/libcuda.so
echo "CUDA root ready: $R ($(ls lib64 | wc -l) entries in lib64)"
