#!/bin/bash
# Track B: BridgeWalk seed 1 (GPU 3 only).
cd ~/research/grag; source waitgpu.sh
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CH=4
L=logs_bridge; DS="musique 2wikimultihopqa hotpotqa"
waitgpu 3000; $PY code/bridgewalk.py train ckpt/joint_s0 ckpt/bridge_s1 1 > $L/train_s1.txt 2>&1
for ds in $DS; do waitgpu 3500; $PY code/bridgewalk.py infer $ds ckpt/bridge_s1 bridges1_full full; done > $L/inf_s1.txt 2>&1
while [ ! -f $L/qa_full.done ]; do sleep 300; done
for ds in $DS; do waitgpu 12500; GMU=0.45 flock .reader.lock $PY code/reader.py $ds bridges1_full_SB300 bridges1_full_SB450 bridges1_full_SB600; done > $L/qa_s1.txt 2>&1
echo ok > $L/all_B.done
