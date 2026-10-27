#!/bin/bash
# StructChain (v6) full pipeline on GPU 6 and GPU 7. All readers int4 (AWQ / bnb), all budgets strict (reader tokenizer).
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1
L=logs_v6; mkdir -p $L
DS="musique 2wikimultihopqa hotpotqa"
# ---------- phase 0: data (GPU 7 train data | GPU 6 test data + no-synonymy proposer) | CPU strict repacking ----------
(CUDA_VISIBLE_DEVICES=7 V6=1 TRAINOUT=chain_train_v6 $PY code/chain_data.py train 12) > $L/data_train.txt 2>&1 &
(export CUDA_VISIBLE_DEVICES=6
 for ds in $DS; do V6=1 CANDSUF=_v6 $PY code/chain_data.py $ds 12; done
 $PY code/train.py --v2 --no_dec --no_diff --no_syn --epochs 6 --seed 0 --out ckpt/gnn_nosyn_s0.pt
 for ds in $DS; do GNN=gnn_nosyn_s0 NOSYN=1 V6=1 CANDSUF=_v6nosyn $PY code/chain_data.py $ds 12; done) > $L/data_test.txt 2>&1 &
(for ds in $DS; do for B in 300 450 600 800; do
   for pair in "dense dense" "hipporag2 hipporag2" "rerankZS_B600 rerankZS" "gnn_s0_B600 gnn" "point_s0_B600 point" "chainDP_s0_B600 chainDP"; do
     set -- $pair; CUDA_VISIBLE_DEVICES="" $PY code/budget.py $ds $1 ${2}_SB$B $B; done; done; done) > $L/repack.txt 2>&1 &
wait
# ---------- phase 1: smoke ----------
CUDA_VISIBLE_DEVICES=7 $PY code/struct_chain.py --limit 24 --out ckpt/struct_smoke > $L/smoke.txt 2>&1 || { echo SMOKE_FAIL > $L/fail; exit 1; }
# ---------- phase 2: training + inference, two lanes ----------
SI="$PY code/struct_infer.py"; ST="$PY code/struct_chain.py"
lane7() { export CUDA_VISIBLE_DEVICES=7
  $ST --seed 0 --out ckpt/structF_s0 > $L/structF_s0.txt 2>&1
  for ds in $DS; do $SI $ds ckpt/structF_s0 structF_s0; $SI $ds ckpt/structF_s0 structF_s0nosyn _v6nosyn; done >> $L/infer.txt 2>&1
  $ST --seed 0 --no_graph --out ckpt/structNG_s0 > $L/structNG_s0.txt 2>&1
  for ds in $DS; do $SI $ds ckpt/structNG_s0 structNG_s0; done >> $L/infer.txt 2>&1
  $ST --seed 0 --no_pair --out ckpt/structNP_s0 > $L/structNP_s0.txt 2>&1
  for ds in $DS; do $SI $ds ckpt/structNP_s0 structNP_s0; done >> $L/infer.txt 2>&1; }
lane6() { export CUDA_VISIBLE_DEVICES=6
  $ST --seed 1 --out ckpt/structF_s1 > $L/structF_s1.txt 2>&1
  for ds in $DS; do $SI $ds ckpt/structF_s1 structF_s1; done >> $L/infer.txt 2>&1
  $ST --seed 0 --no_mark --out ckpt/structNM_s0 > $L/structNM_s0.txt 2>&1
  for ds in $DS; do $SI $ds ckpt/structNM_s0 structNM_s0; done >> $L/infer.txt 2>&1
  CHAINDATA=chain_train_v6 $PY code/chain_train.py --mode pairnb --seed 0 --out ckpt/chainPNB6_s0 > $L/chainPNB6.txt 2>&1
  for ds in $DS; do CANDSUF=_v6 $PY code/chain_infer.py $ds ckpt/chainPNB6_s0 pairnb chainPNB6
    for B in 300 450 600 800; do $PY code/budget.py $ds chainPNB6_B600 chainPNB6_SB$B $B; done; done >> $L/infer.txt 2>&1; }
lane7 & lane6 & wait
echo TRAIN_DONE > $L/train.done
# ---------- phase 3: QA (Qwen2.5-7B AWQ int4), then Llama-3.1-8B int4 ----------
qa() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85; shift
  for ds in "$@"; do M=""
    for B in 300 450 600 800; do M="$M structF_s0_SB$B dense_SB$B hipporag2_SB$B rerankZS_SB$B gnn_SB$B chainPNB6_SB$B"; done
    for B in 450 600; do M="$M structF_s1_SB$B structNG_s0_SB$B structNM_s0_SB$B structNP_s0_SB$B point_SB$B chainDP_SB$B structF_s0nosyn_SB$B"; done
    $PY code/reader.py $ds $M > $L/qa_$ds.txt 2>&1; done; }
qa 7 musique 2wikimultihopqa & qa 6 hotpotqa & wait
ll() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85 RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct; shift
  for ds in "$@"; do mkdir -p results_llama/$ds; M=""
    for m in structF_s0 dense hipporag2 rerankZS gnn chainPNB6; do cp results/$ds/${m}_SB450.json results_llama/$ds/; M="$M ${m}_SB450"; done
    $PY code/reader.py $ds $M > $L/llama_$ds.txt 2>&1; done; }
ll 7 musique 2wikimultihopqa & ll 6 hotpotqa & wait
echo ALL > $L/all.done
