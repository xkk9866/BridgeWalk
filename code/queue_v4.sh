#!/bin/bash
# v4 (granularity-adaptive requirement decoding): two lanes on GPU 6 and GPU 7.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1
T="$PY code/train.py --epochs 6 --v3 --gran"
lane_train() {   # $1 gpu, rest: "name|args"
  g=$1; shift
  for job in "$@"; do n=${job%%|*}; a=${job#*|}
    CUDA_VISIBLE_DEVICES=$g $T $a --out ckpt/$n.pt > logs_rev/$n.txt 2>&1 &
  done; wait; }
lane_train 6 "v4_s0|--seed 0" "v4_s2|--seed 2" "v4_s4|--seed 4" "v4k0_s0|--seed 0 --kappa 0" "v4nr_s0|--seed 0 --no_restart" &
lane_train 7 "v4_s1|--seed 1" "v4_s3|--seed 3" "v4k3_s0|--seed 0 --kappa 3" "v4nr_s1|--seed 1 --no_restart" "v4nr_s2|--seed 2 --no_restart" &
wait
echo TRAIN_DONE > logs_rev/v4train.done

eval_lane() {   # $1 gpu, $2 dataset
  export CUDA_VISIBLE_DEVICES=$1; ds=$2; R="$PY code/run_v3.py"
  $R $ds v4_s0 v4_s1 v4_s2 v4_s3 v4_s4 v4k0_s0 v4k3_s0 v4nr_s0 v4nr_s1 v4nr_s2 >> logs_rev/v4eval_$ds.txt 2>&1
  for B in 300 450 600 800; do
    BUDGET=$B FILL=$B TAG=_B$B $R $ds v4_s0 v4_s1 v4_s2 v4_s3 v4_s4 >> logs_rev/v4eval_$ds.txt 2>&1
    BUDGET=$B TOPK=20 TAG=_B$B $R $ds gnn_s0 gnn_s1 gnn_s2 gnn_s3 gnn_s4 >> logs_rev/v4eval_$ds.txt 2>&1
    BUDGET=$B TAG=_eqB$B $PY code/run_retrieval.py $ds hipporag2 dense >> logs_rev/v4eval_$ds.txt 2>&1
  done; }
(eval_lane 6 musique; eval_lane 6 hotpotqa) &
(eval_lane 7 2wikimultihopqa) &
wait
echo EVAL_DONE > logs_rev/v4eval.done

qa_lane() {   # $1 gpu, $2 dataset
  export CUDA_VISIBLE_DEVICES=$1 GMU=0.85; ds=$2; RD="$PY code/reader.py"
  M=""; for B in 300 450 600 800; do M="$M v4_s0_B$B v4_s1_B$B gnn_s0_B$B gnn_s1_B$B hipporag2_eqB$B dense_eqB$B"; done
  $RD $ds $M > logs_rev/v4qa_budget_$ds.txt 2>&1
  $RD $ds v4_s0 v4_s1 v4_s2 v4_s3 v4_s4 v4k0_s0 v4k3_s0 v4nr_s0 v4nr_s1 > logs_rev/v4qa_main_$ds.txt 2>&1
  for B in 300 450 600 800; do echo; done
  $RD $ds v4_s2_B450 v4_s3_B450 v4_s4_B450 gnn_s2_B450 gnn_s3_B450 gnn_s4_B450 v4_s2_B600 v4_s3_B600 v4_s4_B600 gnn_s2_B600 gnn_s3_B600 gnn_s4_B600 > logs_rev/v4qa_seeds_$ds.txt 2>&1; }
(qa_lane 6 musique; qa_lane 6 hotpotqa) &
(qa_lane 7 2wikimultihopqa; for ds in 2wikimultihopqa hotpotqa; do mkdir -p results_llama/$ds; for m in v4_s0_B450 v4_s0_B600 gnn_s0_B450 gnn_s0_B600 hipporag2_eqB450 hipporag2_eqB600; do cp results/$ds/$m.json results_llama/$ds/ 2>/dev/null; done;
   while [ ! -f results/$ds/gnn_s0_B600.json ]; do sleep 60; done
   for m in v4_s0_B450 v4_s0_B600 gnn_s0_B450 gnn_s0_B600 hipporag2_eqB450 hipporag2_eqB600; do cp results/$ds/$m.json results_llama/$ds/; done
   CUDA_VISIBLE_DEVICES=7 GMU=0.85 RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct $PY code/reader.py $ds v4_s0_B450 v4_s0_B600 gnn_s0_B450 gnn_s0_B600 hipporag2_eqB450 hipporag2_eqB600 > logs_rev/v4qa_llama_$ds.txt 2>&1; done) &
wait
echo ALL > logs_rev/v4all.done
