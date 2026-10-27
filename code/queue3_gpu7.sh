#!/bin/bash
# Pareto control: supervised GNN score-top-k at k=2,3,4 vs UniCoR adaptive decoding (equal-token comparison).
cd ~/research/grag
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PY=.venv/bin/python
while [ ! -f logs_rev/all2.done ]; do sleep 120; done
for ds in musique 2wikimultihopqa hotpotqa; do
  for k in 2 3 4; do TOPK=$k TAG=_top$k $PY code/run_v3.py $ds gnn_s0 >> logs_rev/pareto_$ds.txt 2>&1; done
  GMU=0.80 $PY code/reader.py $ds gnn_s0_top2 gnn_s0_top3 gnn_s0_top4 gnn_s0 > logs_rev/qa3_pareto_$ds.txt 2>&1
done
$PY code/aggregate.py > logs_rev/aggregate.txt 2>&1
echo ALL3 > logs_rev/all3.done
