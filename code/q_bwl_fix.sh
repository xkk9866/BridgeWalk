#!/bin/bash
# relaunch the local test walks that ran out of memory beside the mining lanes, once mining is finished (GPU 4)
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs_cover
while [ ! -f $L/mine.done ]; do sleep 30; done
for ds in musique hotpotqa; do for s in 0 1; do SHARD=$s NSHARD=2 $PY code/coverwalk.py infer $ds ckpt/bridge2_s0 bwl local > $L/inf_bwl_${ds}_$s.txt 2>&1 & sleep 20; done; done
wait
