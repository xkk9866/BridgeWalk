#!/bin/bash
# Cross-fitted CoverWalk (GPU 5): half-models -> mining on the held-out half -> global training -> test walks -> packing
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CH=4
L=logs_cross; mkdir -p $L; DS="musique 2wikimultihopqa hotpotqa"
for f in 0 1; do (FOLD=$f $PY code/bridgewalk.py train ckpt/joint_s0 ckpt/half$f 0 > $L/train_half$f.txt 2>&1) & sleep 30; done
wait
# model trained on fold f mines fold 1-f
for f in 0 1; do for ds in $DS; do for s in 0 1; do
  HOLD=$((1-f)) MINEOUT=ckpt/cw_mine_x $PY code/coverwalk.py mine $ds ckpt/half$f $s 2 > $L/mine_${f}_${ds}_$s.txt 2>&1 & sleep 15
done; done; done
wait
echo ok > $L/mine.done
for sd in 0 1; do (MINE=ckpt/cw_mine_x CH=16 $PY code/coverwalk.py train ckpt/bridge2_s0 ckpt/crossw_s$sd $sd > $L/train_cw_s$sd.txt 2>&1) & sleep 30; done
wait
echo ok > $L/train.done
for ds in $DS; do for s in 0 1 2; do SHARD=$s NSHARD=3 $PY code/coverwalk.py infer $ds ckpt/crossw_s0 xw global > $L/inf_xw_${ds}_$s.txt 2>&1 & sleep 15; done; done
wait
for ds in $DS; do $PY code/coverwalk.py pack $ds xw mbr; $PY code/coverwalk.py pack $ds xw marg; done > $L/pack_xw.txt 2>&1
echo ok > $L/inf.done
