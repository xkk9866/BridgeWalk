#!/bin/bash
# Full pipeline on server GPU 1 only.
cd ~/research/grag
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PY=.venv/bin/python
set -x
$PY code/openie_vllm.py || exit 1
$PY code/index.py data/musique data/2wikimultihopqa data/hotpotqa data/train/hotpotqa data/train/2wikimultihopqa || exit 1
$PY code/train.py --out ckpt/unicor.pt || exit 1
for ds in musique 2wikimultihopqa hotpotqa; do $PY code/run_retrieval.py $ds bm25 dense kg_only fixed_fusion rrf hipporag hipporag2 unicor; done
echo PIPELINE_DONE
