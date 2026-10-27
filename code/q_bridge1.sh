#!/bin/bash
# BridgeWalk full pipeline on GPU 3 only (shared server: every stage waits for free memory).
cd ~/research/grag; source waitgpu.sh
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false RB=8 CH=4 IBS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs_bridge; DS="musique 2wikimultihopqa hotpotqa"


waitgpu 2400; $PY code/bridgewalk.py train ckpt/joint_s0 ckpt/bridge_s0 0 > $L/train_s0.txt 2>&1
waitgpu 2400;  for ds in $DS mhrag; do $PY code/bridgewalk.py infer $ds ckpt/bridge_s0 bridge_full full; done > $L/inf_full.txt 2>&1
echo ok > $L/inf_full.done
for v in nostop qbridge oie nograph; do waitgpu 2400; for ds in $DS; do $PY code/bridgewalk.py infer $ds ckpt/bridge_s0 bridge_$v $v; done; done > $L/inf_abl.txt 2>&1
qa() { waitgpu 21500; GMU=0.85 $PY code/reader.py "$@"; }
for ds in $DS; do M=""; for B in 300 450 600; do M="$M bridge_full_SB$B"; done
  for v in nostop qbridge oie nograph; do M="$M bridge_${v}_SB450"; done; qa $ds $M; done > $L/qa.txt 2>&1
qa mhrag bridge_full_SB300 bridge_full_SB450 bridge_full_SB600 >> $L/qa.txt 2>&1
echo ok > $L/qa.done
waitgpu 2400; $PY code/bridgewalk.py train ckpt/joint_s0 ckpt/bridge_s1 1 > $L/train_s1.txt 2>&1
waitgpu 2400;  for ds in $DS; do $PY code/bridgewalk.py infer $ds ckpt/bridge_s1 bridges1_full full; done > $L/inf_s1.txt 2>&1
for ds in $DS; do qa $ds bridges1_full_SB300 bridges1_full_SB450 bridges1_full_SB600; done > $L/qa_s1.txt 2>&1
export RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct
for ds in $DS; do mkdir -p results_llama/$ds; cp results/$ds/bridge_full_SB450.json results_llama/$ds/; qa $ds bridge_full_SB450; done > $L/llama.txt 2>&1
unset RESDIR READER
$PY code/final_stats.py > $L/stats.txt 2>&1; $PY code/walk_ret.py >> $L/stats.txt 2>&1
echo ok > $L/all.done
