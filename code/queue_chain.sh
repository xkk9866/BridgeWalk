#!/bin/bash
# ChainCoR: graph-proposed, KG-bridged chain verification. Two lanes: GPU 6 and GPU 7. Readers are int4 (AWQ / bnb).
cd ~/research/grag
pkill -f "queue_v4.s[h]"
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1
mkdir -p logs_chain
CUDA_VISIBLE_DEVICES=7 $PY code/chain_data.py train 12 > logs_chain/data_train.txt 2>&1 &
(for ds in musique 2wikimultihopqa hotpotqa; do CUDA_VISIBLE_DEVICES=6 $PY code/chain_data.py $ds 12; done) > logs_chain/data_test.txt 2>&1 &
wait
TR="$PY code/chain_train.py"
CUDA_VISIBLE_DEVICES=7 $TR --mode pair --seed 0 --out ckpt/chain_pair_s0 > logs_chain/pair_s0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 $TR --mode pairnb --seed 0 --out ckpt/chain_pairnb_s0 > logs_chain/pairnb_s0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 $TR --mode pair --seed 1 --out ckpt/chain_pair_s1 > logs_chain/pair_s1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 $TR --mode point --seed 0 --out ckpt/chain_point_s0 > logs_chain/point_s0.txt 2>&1 &
wait
IN="$PY code/chain_infer.py"
infer_lane() { export CUDA_VISIBLE_DEVICES=$1; shift
  for ds in "$@"; do
    $IN $ds ckpt/chain_pair_s0 pair chainP_s0 >> logs_chain/infer_$ds.txt 2>&1
    $IN $ds ckpt/chain_pair_s1 pair chainP_s1 >> logs_chain/infer_$ds.txt 2>&1
    $IN $ds ckpt/chain_pairnb_s0 pairnb chainNB_s0 >> logs_chain/infer_$ds.txt 2>&1
    $IN $ds ckpt/chain_point_s0 point point_s0 >> logs_chain/infer_$ds.txt 2>&1
    $IN $ds zeroshot point rerankZS >> logs_chain/infer_$ds.txt 2>&1
  done; }
infer_lane 7 musique 2wikimultihopqa &
infer_lane 6 hotpotqa &
wait
qa_lane() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85; shift
  for ds in "$@"; do
    M="bm25 dense kg_only fixed_fusion rrf hipporag hipporag2"
    for B in 300 450 600 800; do M="$M chainP_s0_B$B chainP_s0_fB$B gnn_s0_B$B hipporag2_eqB$B dense_eqB$B rerankZS_B$B"; done
    for B in 450 600; do M="$M chainP_s1_B$B chainP_s1_fB$B chainNB_s0_B$B point_s0_B$B gnn_s1_B$B"; done
    $PY code/reader.py $ds $M > logs_chain/qa_$ds.txt 2>&1
  done; }
qa_lane 7 musique 2wikimultihopqa &
qa_lane 6 hotpotqa &
wait
llama_lane() { export CUDA_VISIBLE_DEVICES=$1 GMU=0.85 RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct; shift
  for ds in "$@"; do mkdir -p results_llama/$ds
    for m in chainP_s0_B600 chainP_s0_fB600 gnn_s0_B600 hipporag2_eqB600 dense_eqB600 rerankZS_B600; do cp results/$ds/$m.json results_llama/$ds/; done
    $PY code/reader.py $ds chainP_s0_B600 chainP_s0_fB600 gnn_s0_B600 hipporag2_eqB600 dense_eqB600 rerankZS_B600 > logs_chain/llama_$ds.txt 2>&1
  done; }
llama_lane 7 musique 2wikimultihopqa &
llama_lane 6 hotpotqa &
wait
echo ALL > logs_chain/all.done
