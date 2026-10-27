#!/bin/bash
# BridgeWalk on the 410k-passage open corpus, GPU 1, after the matched model has finished training.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs_ctrl
until grep -q "^saved" $L/train_fixnb.txt; do sleep 60; done
for ds in musique 2wikimultihopqa hotpotqa; do rm -f results/open_$ds/bw.walks*.json
  for s in 0 1; do CUDA_VISIBLE_DEVICES=1 SHARD=$s NSHARD=2 $PY code/coverwalk.py infer open_$ds ckpt/bridge2_s0 bw local > $L/inf_bw_open_${ds}_$s.txt 2>&1 & sleep 20; done; done
wait
for ds in musique 2wikimultihopqa hotpotqa; do $PY code/coverwalk.py pack open_$ds bw marg; done > $L/pack_open.txt 2>&1
for ds in musique 2wikimultihopqa hotpotqa; do CUDA_VISIBLE_DEVICES=1 GMU=0.6 $PY code/reader.py open_$ds bw_marg_SB450 bw_marg_SB600 walk_full_SB450 walk_full_SB600; done > $L/qa_open.txt 2>&1
echo ok > $L/open.done
