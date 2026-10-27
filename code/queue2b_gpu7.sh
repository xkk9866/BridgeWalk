#!/bin/bash
# Fixed recovery queue: wait for the 3 retrains (already running), evaluate them, then all QA with one int4 reader.
cd ~/research/grag
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 GMU=0.70
RD="$PY code/reader.py"
BASE="bm25 dense kg_only fixed_fusion rrf hipporag hipporag2 dense_ft"
MAIN=""; for s in 0 1 2; do MAIN="$MAIN unicor3_s$s unicor3_s${s}_fill u3nr_s$s gnn_s${s}_fill"; done
for ds in musique 2wikimultihopqa hotpotqa; do $RD $ds $BASE > logs_rev/qa2_base_$ds.txt 2>&1; done
for ds in musique hotpotqa; do $RD $ds $MAIN > logs_rev/qa2_main_$ds.txt 2>&1; done
while pgrep -f "code/train.p[y]" > /dev/null; do sleep 60; done
for ds in musique 2wikimultihopqa hotpotqa; do
  $PY code/run_v3.py $ds u3tf_s0 u3noconf_s0 u3soft_s0 >> logs_rev/eval2_$ds.txt 2>&1
  INJECT=1 TAG=_inj $PY code/run_v3.py $ds u3noconf_s0 >> logs_rev/inject2_$ds.txt 2>&1
done
for ds in musique 2wikimultihopqa hotpotqa; do
  $RD $ds u3nodiff_s0 u3noctrl_s0 u3noprov_s0 u3tf_s0 u3noconf_s0 u3soft_s0 pointer_s0 > logs_rev/qa2_abl_$ds.txt 2>&1
  $RD $ds unicor3_s0_suff0.3 unicor3_s0_suff0.7 unicor3_s0_suff0.9 unicor3_s0_fix3 unicor3_s0_fix5 unicor3_s0_thr0.2 > logs_rev/qa2_stop_$ds.txt 2>&1
done
for ds in 2wikimultihopqa hotpotqa; do
  RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct $RD $ds dense hipporag2 dense_ft gnn_s0_fill unicor3_s0 unicor3_s0_fill > logs_rev/qa2_llama_$ds.txt 2>&1
done
$PY code/aggregate.py > logs_rev/aggregate.txt 2>&1
echo ALL2 > logs_rev/all2.done
