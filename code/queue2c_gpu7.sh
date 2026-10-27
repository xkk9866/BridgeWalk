#!/bin/bash
# Trimmed QA queue (same protocol). Takes over from queue2b after its current reader process ends.
cd ~/research/grag
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 GMU=0.70
RD="$PY code/reader.py"
while pgrep -f "code/reader.p[y]" > /dev/null; do sleep 30; done
MAIN="unicor3_s0 unicor3_s0_fill u3nr_s0 gnn_s0_fill unicor3_s1 unicor3_s1_fill u3nr_s1 gnn_s1_fill"
for ds in musique hotpotqa; do $RD $ds $MAIN > logs_rev/qa2_main_$ds.txt 2>&1; done
while pgrep -f "code/train.p[y]" > /dev/null; do sleep 60; done
export GMU=0.85
for ds in musique 2wikimultihopqa hotpotqa; do
  $PY code/run_v3.py $ds u3tf_s0 u3noconf_s0 u3soft_s0 >> logs_rev/eval2_$ds.txt 2>&1
  INJECT=1 TAG=_inj $PY code/run_v3.py $ds u3noconf_s0 >> logs_rev/inject2_$ds.txt 2>&1
  for k in 2 3 4; do TOPK=$k TAG=_top$k $PY code/run_v3.py $ds gnn_s0 >> logs_rev/pareto_$ds.txt 2>&1; done
done
for ds in musique 2wikimultihopqa hotpotqa; do
  $RD $ds gnn_s0_top2 gnn_s0_top3 gnn_s0_top4 gnn_s0 unicor3_s0_suff0.3 unicor3_s0_suff0.7 unicor3_s0_fix3 > logs_rev/qa3_pareto_$ds.txt 2>&1
  $RD $ds u3tf_s0 u3soft_s0 u3nodiff_s0 pointer_s0 > logs_rev/qa2_abl_$ds.txt 2>&1
done
for ds in 2wikimultihopqa hotpotqa; do
  RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct $RD $ds dense hipporag2 dense_ft gnn_s0_fill unicor3_s0 unicor3_s0_fill > logs_rev/qa2_llama_$ds.txt 2>&1
done
$PY code/aggregate.py > logs_rev/aggregate.txt 2>&1
echo ALL > logs_rev/all3.done
