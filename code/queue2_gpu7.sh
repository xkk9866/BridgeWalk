#!/bin/bash
# Recovery queue (GPU 7 only): retrain 3 killed runs in parallel with a single int4 vLLM reader.
cd ~/research/grag
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1
T="$PY code/train.py --epochs 6 --seed 0"
(
  $T --v3 --tf --out ckpt/u3tf_s0.pt > logs_rev/u3tf_s0.txt 2>&1 &
  $T --v3 --no_conf --out ckpt/u3noconf_s0.pt > logs_rev/u3noconf_s0.txt 2>&1 &
  $T --v3 --soft_prov --out ckpt/u3soft_s0.pt > logs_rev/u3soft_s0.txt 2>&1 &
  wait
  for ds in musique 2wikimultihopqa hotpotqa; do
    $PY code/run_v3.py $ds u3tf_s0 u3noconf_s0 u3soft_s0 >> logs_rev/eval2_$ds.txt 2>&1
    INJECT=1 TAG=_inj $PY code/run_v3.py $ds u3noconf_s0 >> logs_rev/inject2_$ds.txt 2>&1
  done
  echo done > logs_rev/retrain.done
) &
RD="GMU=0.70 $PY code/reader.py"
BASE="bm25 dense kg_only fixed_fusion rrf hipporag hipporag2 dense_ft"
MAIN=""; for s in 0 1 2; do MAIN="$MAIN unicor3_s$s unicor3_s${s}_fill u3nr_s$s gnn_s${s}_fill"; done
for ds in musique 2wikimultihopqa hotpotqa; do $RD $ds $BASE > logs_rev/qa2_base_$ds.txt 2>&1; done
for ds in musique hotpotqa; do $RD $ds $MAIN > logs_rev/qa2_main_$ds.txt 2>&1; done
while [ ! -f logs_rev/retrain.done ]; do sleep 60; done
for ds in musique 2wikimultihopqa hotpotqa; do
  $RD $ds u3nodiff_s0 u3noctrl_s0 u3noprov_s0 u3tf_s0 u3noconf_s0 u3soft_s0 pointer_s0 > logs_rev/qa2_abl_$ds.txt 2>&1
  $RD $ds unicor3_s0_suff0.3 unicor3_s0_suff0.7 unicor3_s0_suff0.9 unicor3_s0_fix3 unicor3_s0_fix5 unicor3_s0_thr0.2 > logs_rev/qa2_stop_$ds.txt 2>&1
done
for ds in 2wikimultihopqa hotpotqa; do
  RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct $RD $ds dense hipporag2 dense_ft gnn_s0_fill unicor3_s0 unicor3_s0_fill > logs_rev/qa2_llama_$ds.txt 2>&1
done
$PY code/aggregate.py > logs_rev/aggregate.txt 2>&1
echo ALL2 > logs_rev/all2.done
