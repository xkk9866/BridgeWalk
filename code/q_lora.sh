#!/bin/bash
# LoRA vs full fine-tuning at small scale (same 2,000 training groups, same seed); evaluated on 300 MuSiQue questions.
cd ~/research/grag; source waitgpu.sh
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CH=4
L=logs_bridge
while [ ! -f $L/train2.done ]; do sleep 600; done
for t in 1 2 3; do waitgpu 2500; SMOKE=2000 $PY code/bridgewalk.py train ckpt/joint_s0 ckpt/bw_lora2k 0 > $L/lora2k.txt 2>&1 && break; sleep 300; done
for t in 1 2 3; do waitgpu 9000; FULLFT=1 SMOKE=2000 $PY code/bridgewalk.py train ckpt/joint_s0 ckpt/bw_full2k 0 > $L/full2k.txt 2>&1 && break; sleep 600; done
for v in lora2k full2k; do waitgpu 3000; LIMIT=300 $PY code/bridgewalk.py infer musique ckpt/bw_$v bw_$v full; done > $L/lora_vs_full.txt 2>&1
echo ok > $L/lora.done
