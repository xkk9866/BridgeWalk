#!/bin/bash
# Reviewer round, restart with fewer lanes per GPU (GPUs 0 and 1 only).
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs_ctrl; DS="musique 2wikimultihopqa hotpotqa"
lane() { g=$1; ds=$2; md=$3; nm=$4; n=$5; shift 5
  for s in $(seq 0 $((n-1))); do env "$@" CUDA_VISIBLE_DEVICES=$g SHARD=$s NSHARD=$n $PY code/coverwalk.py infer $ds $md $nm local > $L/inf_${nm}_${ds}_$s.txt 2>&1 & sleep 20; done; }
ndone() { cat "$@" 2>/dev/null | grep -c "^done"; }
# on-policy rows lack the src tag if built before this change: tag them
$PY - <<'P'
import json
f="ckpt/bridge_onpol.jsonl"; R=[json.loads(l) for l in open(f)]
open(f,"w").write("".join(json.dumps({**r,"src":"onpol"})+"\n" for r in R))
P
(CUDA_VISIBLE_DEVICES=0 CH=16 TEACH_SUB=3000 TRAINF=bridge_train.jsonl,bridge_onpol.jsonl $PY code/bridgewalk.py train ckpt/bridge2_s0 ckpt/onpol_s0 0 > $L/train_onpol.txt 2>&1; echo done >> $L/train_onpol.txt) &
sleep 60
for ds in 2wikimultihopqa hotpotqa; do lane 0 $ds ckpt/bridge2_s0 ctl_d3 2 DEPTH=3; done
for ds in $DS; do lane 0 open_$ds ckpt/bridge2_s0 bw 1; done
# DAgger model: walks on both GPUs once trained
until grep -q "^done" $L/train_onpol.txt; do sleep 60; done
for ds in musique 2wikimultihopqa; do lane 0 $ds ckpt/onpol_s0 onpol 2; done
for ds in hotpotqa mhrag; do lane 1 $ds ckpt/onpol_s0 onpol 2; done
# matched model trained with fixed neighbourhoods
until grep -q "^saved" $L/train_fixnb.txt; do sleep 60; done
for ds in $DS; do lane 1 $ds ckpt/fixnb_s0 fixtr 2 VAR=fixnb; done
wait
echo ok > $L/stage2.done
