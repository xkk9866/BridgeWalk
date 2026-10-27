#!/bin/bash
# v7: grounded evidence graph with subject links -> proposer -> pairwise verifier -> Prim ordering -> strict packing.
# GPU 6 and GPU 7 only; readers int4.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1
L=logs_v7; mkdir -p $L
DS="musique 2wikimultihopqa hotpotqa"
# baseline for the ablation "no subject links": the v6-data pairwise verifier with Prim ordering + strict packing
(CUDA_VISIBLE_DEVICES=7; for ds in $DS; do CUDA_VISIBLE_DEVICES=7 $PY code/chain_infer2.py $ds ckpt/chainPNB6_s0 chainNS $(echo _v6); done) > $L/infer_nosubj.txt 2>&1 &
# proposer with subject links (builds the new training cache on first use)
(export CUDA_VISIBLE_DEVICES=6 SUBJ=1 MAXP=128 CACHESUF=_subj
 $PY code/train.py --v2 --no_dec --no_diff --epochs 6 --seed 0 --out ckpt/gnnS_s0.pt > $L/gnnS_s0.txt 2>&1) &
wait
(export CUDA_VISIBLE_DEVICES=7 SUBJ=1 MAXP=128 CACHESUF=_subj
 $PY code/train.py --v2 --no_dec --no_diff --epochs 6 --seed 1 --out ckpt/gnnS_s1.pt > $L/gnnS_s1.txt 2>&1) &
(export CUDA_VISIBLE_DEVICES=6 SUBJ=1 MAXP=128 GNN=gnnS_s0
 for ds in $DS; do CANDSUF=_v7 $PY code/chain_data.py $ds 12; done > $L/data_test.txt 2>&1
 TRAINOUT=chain_train_v7 $PY code/chain_data.py train 12 > $L/data_train.txt 2>&1) &
wait
(CUDA_VISIBLE_DEVICES=6 CHAINDATA=chain_train_v7 $PY code/chain_train.py --mode pairnb --seed 0 --out ckpt/chainS_s0 > $L/chainS_s0.txt 2>&1) &
(CUDA_VISIBLE_DEVICES=7 CHAINDATA=chain_train_v7 $PY code/chain_train.py --mode pairnb --seed 1 --out ckpt/chainS_s1 > $L/chainS_s1.txt 2>&1) &
wait
(for ds in $DS; do CUDA_VISIBLE_DEVICES=6 $PY code/chain_infer2.py $ds ckpt/chainS_s0 chainS_s0 _v7; done) > $L/infer.txt 2>&1 &
(for ds in $DS; do CUDA_VISIBLE_DEVICES=7 $PY code/chain_infer2.py $ds ckpt/chainS_s1 chainS_s1 _v7; done) > $L/infer_s1.txt 2>&1 &
wait
echo INFER > $L/infer.done
qa() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85; shift
  for ds in "$@"; do M=""
    for B in 300 450 600 800; do M="$M chainS_s0_SB$B chainS_s0_prop_SB$B"; done
    for B in 450 600; do M="$M chainS_s1_SB$B chainNS_SB$B"; done
    $PY code/reader.py $ds $M > $L/qa_$ds.txt 2>&1; done; }
qa 7 musique 2wikimultihopqa & qa 6 hotpotqa & wait
ll() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85 RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct; shift
  for ds in "$@"; do mkdir -p results_llama/$ds; M=""
    for m in chainS_s0 dense hipporag2 rerankZS gnn chainPNB6 chainS_s0_prop; do cp results/$ds/${m}_SB450.json results_llama/$ds/; M="$M ${m}_SB450"; done
    $PY code/reader.py $ds $M > $L/llama_$ds.txt 2>&1; done; }
ll 7 musique 2wikimultihopqa & ll 6 hotpotqa & wait
echo ALL > $L/all.done
