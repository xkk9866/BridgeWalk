#!/bin/bash
# Revision 9: QRET (joint verifier + question-rooted MST), ordering controls, trained MDR, hybrid, news-domain transfer.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 MAXP=128
L=logs_rev9; mkdir -p $L
DS="musique 2wikimultihopqa hotpotqa"
# ---- lane 7: joint verifier seed 0, then QRET inference ----
lane7() { export CUDA_VISIBLE_DEVICES=7
  CHAINDATA=chain_train_v7 $PY code/chain_train.py --mode joint --seed 0 --out ckpt/joint_s0 > $L/joint_s0.txt 2>&1
  while [ ! -f $L/mh_cand.done ]; do sleep 60; done
  for ds in $DS mhrag; do $PY code/chain_infer3.py $ds ckpt/joint_s0 joint_s0 _v7; done > $L/infer_joint_s0.txt 2>&1
  $PY code/chain_infer2.py mhrag ckpt/chainS_s0 chainS_s0 _v7 > $L/mh_chainS.txt 2>&1; }
# ---- lane 6: news-domain data, trained MDR + hybrid, joint verifier seed 1 ----
lane6() { export CUDA_VISIBLE_DEVICES=6
  $PY code/openie_vllm.py mhrag > $L/mh_openie.txt 2>&1
  $PY code/index.py data/mhrag > $L/mh_index.txt 2>&1
  $PY code/run_retrieval.py mhrag bm25 dense hipporag2 > $L/mh_base.txt 2>&1
  GNN=gnnS_s0 CANDSUF=_v7 $PY code/chain_data.py mhrag 12 >> $L/mh_base.txt 2>&1
  CANDSUF=_v7 $PY code/chain_infer.py mhrag zeroshot point rerankZS >> $L/mh_base.txt 2>&1
  CANDSUF=_v7 $PY code/chain_infer.py mhrag ckpt/chain_point_s0 point point_s0 >> $L/mh_base.txt 2>&1
  echo ok > $L/mh_cand.done
  $PY code/mdr_train.py > $L/mdr_train.txt 2>&1
  for ds in $DS mhrag; do $PY code/strong_baselines.py $ds; done > $L/strong.txt 2>&1
  CHAINDATA=chain_train_v7 $PY code/chain_train.py --mode joint --seed 1 --out ckpt/joint_s1 > $L/joint_s1.txt 2>&1
  for ds in $DS; do $PY code/chain_infer3.py $ds ckpt/joint_s1 joint_s1 _v7; done > $L/infer_joint_s1.txt 2>&1; }
lane7 & lane6 & wait
# ---- strict re-packing of all ranking-only methods ----
for ds in $DS mhrag; do for B in 300 450 600 800; do
  for m in mdrT hybrid; do $PY code/budget.py $ds $m ${m}_SB$B $B; done; done; done > $L/repack.txt 2>&1
for ds in $DS; do for B in 300 450 600 800; do $PY code/budget.py $ds mdr mdr_SB$B $B; done; done >> $L/repack.txt 2>&1
for B in 300 450 600 800; do for pair in "dense dense" "hipporag2 hipporag2" "rerankZS_B600 rerankZS" "point_s0_B600 point"; do
  set -- $pair; $PY code/budget.py mhrag $1 ${2}_SB$B $B; done; done >> $L/repack.txt 2>&1
echo REPACK > $L/repack.done
# ---- QA (Qwen AWQ int4) ----
qa() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85; shift
  for ds in "$@"; do M=""
    if [ $ds = mhrag ]; then
      for B in 300 450 600; do M="$M joint_s0_qret_SB$B chainS_s0_SB$B chainS_s0_prop_SB$B dense_SB$B hipporag2_SB$B rerankZS_SB$B point_SB$B mdrT_SB$B hybrid_SB$B"; done
    else
      for B in 300 450 600 800; do M="$M joint_s0_qret_SB$B mdrT_SB$B hybrid_SB$B"; done
      for B in 300 450 600; do M="$M joint_s0_mst_SB$B joint_s0_greedy_SB$B joint_s0_point_SB$B joint_s0_mmr_SB$B joint_s0_cover_SB$B"; done
      for B in 450 600; do M="$M joint_s1_qret_SB$B"; done
    fi
    $PY code/reader.py $ds $M > $L/qa_$ds.txt 2>&1; done; }
qa 7 musique 2wikimultihopqa & qa 6 hotpotqa mhrag & wait
# ---- second reader (Llama-3.1-8B int4) at B=450 ----
ll() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85 RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct; shift
  for ds in "$@"; do mkdir -p results_llama/$ds; M=""
    for m in joint_s0_qret mdrT hybrid; do cp results/$ds/${m}_SB450.json results_llama/$ds/; M="$M ${m}_SB450"; done
    $PY code/reader.py $ds $M > $L/llama_$ds.txt 2>&1; done; }
ll 7 musique 2wikimultihopqa & ll 6 hotpotqa & wait
$PY code/final_stats.py > $L/stats.txt 2>&1
echo ALL > $L/all.done
