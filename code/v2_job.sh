#!/bin/bash
# waits for OpenIE of training pools, then index + train v2 (GPU1) and its key ablations (GPU2)
cd ~/research/grag; export HF_HUB_OFFLINE=1 PY=.venv/bin/python
while pgrep -f "openie_vll[m]|reade[r].py" >/dev/null; do sleep 60; done
# hotpot/2wiki incremental OpenIE if not yet done by GPU2
CUDA_VISIBLE_DEVICES=2 $PY code/openie_vllm.py train/hotpotqa train/2wikimultihopqa
CUDA_VISIBLE_DEVICES=1 $PY code/index.py data/train/musique data/train/hotpotqa data/train/2wikimultihopqa || exit 1
CUDA_VISIBLE_DEVICES=1 $PY code/train.py --v2 --epochs 8 --out ckpt/unicor2.pt > log_v2.txt 2>&1 &
sleep 900   # let the example cache be written once
CUDA_VISIBLE_DEVICES=2 $PY code/train.py --v2 --no_restart --epochs 8 --out ckpt/unicor2_norestart.pt > log_v2_nr.txt 2>&1
CUDA_VISIBLE_DEVICES=2 $PY code/train.py --v2 --no_diff --epochs 8 --out ckpt/unicor2_nodiff.pt > log_v2_nd.txt 2>&1
wait
for ds in musique 2wikimultihopqa hotpotqa; do CUDA_VISIBLE_DEVICES=1 WIDE=1 $PY code/run_retrieval.py $ds unicor2 unicor2_norestart unicor2_nodiff; done
echo V2_DONE
