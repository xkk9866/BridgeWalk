#!/bin/bash
# Revision 8: missing baselines (pointwise CE at all budgets, MDR-style), KG-edge-constrained comparison. GPU 6/7, int4 readers.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1
L=logs_rev8; mkdir -p $L
DS="musique 2wikimultihopqa hotpotqa"
(export CUDA_VISIBLE_DEVICES=6
 for ds in $DS; do $PY code/mdr_style.py $ds
   for B in 300 450 600 800; do $PY code/budget.py $ds mdr mdr_SB$B $B; done
   for B in 300 800; do $PY code/budget.py $ds point_s0_B600 point_SB$B $B; done; done) > $L/base.txt 2>&1 &
(export CUDA_VISIBLE_DEVICES=7
 for ds in $DS; do KGMASK=1 $PY code/chain_infer2.py $ds ckpt/chainS_s0 chainKG _v7; done) > $L/kgmask.txt 2>&1 &
wait
qa() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85; shift
  for ds in "$@"; do
    $PY code/reader.py $ds point_SB300 point_SB800 mdr_SB300 mdr_SB450 mdr_SB600 mdr_SB800 chainKG_SB450 chainKG_SB600 > $L/qa_$ds.txt 2>&1; done; }
qa 7 musique 2wikimultihopqa & qa 6 hotpotqa & wait
ll() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85 RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct; shift
  for ds in "$@"; do for m in point_SB450 mdr_SB450; do cp results/$ds/$m.json results_llama/$ds/; done
    $PY code/reader.py $ds point_SB450 mdr_SB450 > $L/llama_$ds.txt 2>&1; done; }
ll 7 musique 2wikimultihopqa & ll 6 hotpotqa & wait
echo ALL > $L/all.done
