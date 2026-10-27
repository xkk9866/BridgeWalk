#!/bin/bash
# Reviewer round: matched expansion control, depth control, expansion ablations, open corpus, on-policy training.
# Server GPUs 0 and 1 only.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs_ctrl; mkdir -p $L; DS="musique 2wikimultihopqa hotpotqa"
# lane: <gpu> <ds> <model> <name> <nshard> [env...]
lane() { g=$1; ds=$2; md=$3; nm=$4; n=$5; shift 5
  for s in $(seq 0 $((n-1))); do env "$@" CUDA_VISIBLE_DEVICES=$g SHARD=$s NSHARD=$n $PY code/coverwalk.py infer $ds $md $nm local > $L/inf_${nm}_${ds}_$s.txt 2>&1 & sleep 15; done; }

# training data (CPU + query embedding), then the two trainings
(CUDA_VISIBLE_DEVICES=0 $PY code/bridgewalk.py build_onpolicy ckpt/cw_mine_x bridge_onpol.jsonl > $L/build_onpol.txt 2>&1
 CUDA_VISIBLE_DEVICES=0 CH=16 TRAINF=bridge_train.jsonl,bridge_onpol.jsonl $PY code/bridgewalk.py train ckpt/joint_s0 ckpt/onpol_s0 0 > $L/train_onpol.txt 2>&1) &
(CUDA_VISIBLE_DEVICES=1 NEG=fixnb TRAINF=bridge_train_fixnb.jsonl $PY code/bridgewalk.py build > $L/build_fixnb.txt 2>&1
 CUDA_VISIBLE_DEVICES=1 CH=16 TRAINF=bridge_train_fixnb.jsonl $PY code/bridgewalk.py train ckpt/joint_s0 ckpt/fixnb_s0 0 > $L/train_fixnb.txt 2>&1) &
sleep 60
# controls with the same BridgeWalk weights
for ds in $DS; do lane 0 $ds ckpt/bridge2_s0 ctl_fixnb 2 VAR=fixnb; done
for ds in $DS; do lane 0 $ds ckpt/bridge2_s0 ctl_d3 2 DEPTH=3; done
for ds in $DS; do lane 1 $ds ckpt/bridge2_s0 ctl_allment 2 VAR=allment; done
lane 1 musique ckpt/bridge2_s0 ctl_oie 2 VAR=oie
lane 1 musique ckpt/bridge2_s0 ctl_qbridge 1 VAR=qbridge
for ds in $DS; do lane 1 open_$ds ckpt/bridge2_s0 bw 2; done
lane 1 open_musique ckpt/bridge2_s0 ctl_fixnb 2 VAR=fixnb
wait
echo ok > $L/stage1.done
# models trained with their own expansion
for ds in $DS mhrag; do lane 0 $ds ckpt/onpol_s0 onpol 2; done
for ds in $DS; do lane 1 $ds ckpt/fixnb_s0 fixtr 2 VAR=fixnb; done
wait
echo ok > $L/stage2.done
