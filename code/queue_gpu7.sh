#!/bin/bash
# All experiments for the revision, GPU 7 only, as a 4-phase concurrent queue.
cd ~/research/grag
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1
mkdir -p logs_rev ckpt
T="$PY code/train.py --epochs 6"
# ---------------- phase 1: training (10 concurrent) ----------------
{
echo "$PY code/ft_dense.py > logs_rev/ft_dense.txt 2>&1"
for s in 0 1 2 3 4; do
  echo "$T --v3 --seed $s --out ckpt/unicor3_s$s.pt > logs_rev/unicor3_s$s.txt 2>&1"
  echo "$T --v3 --no_restart --seed $s --out ckpt/u3nr_s$s.pt > logs_rev/u3nr_s$s.txt 2>&1"
  echo "$T --v2 --no_dec --no_diff --seed $s --out ckpt/gnn_s$s.pt > logs_rev/gnn_s$s.txt 2>&1"
done
for s in 0 1 2; do
  echo "$T --v3 --no_diff --seed $s --out ckpt/u3nodiff_s$s.pt > logs_rev/u3nodiff_s$s.txt 2>&1"
  echo "$T --v3 --no_red --no_ground --seed $s --out ckpt/u3noctrl_s$s.pt > logs_rev/u3noctrl_s$s.txt 2>&1"
  echo "$T --v3 --no_prov --seed $s --out ckpt/u3noprov_s$s.pt > logs_rev/u3noprov_s$s.txt 2>&1"
  echo "$T --v3 --tf --seed $s --out ckpt/u3tf_s$s.pt > logs_rev/u3tf_s$s.txt 2>&1"
  echo "$T --v3 --no_conf --seed $s --out ckpt/u3noconf_s$s.pt > logs_rev/u3noconf_s$s.txt 2>&1"
  echo "$T --v3 --soft_prov --seed $s --out ckpt/u3soft_s$s.pt > logs_rev/u3soft_s$s.txt 2>&1"
  echo "$T --v2 --no_diff --no_restart --no_red --no_ground --seed $s --out ckpt/pointer_s$s.pt > logs_rev/pointer_s$s.txt 2>&1"
done
for n in 300 1000; do for s in 0 1; do
  echo "$T --v3 --n_train $n --seed $s --out ckpt/u3n${n}_s$s.pt > logs_rev/u3n${n}_s$s.txt 2>&1"
  echo "$T --v2 --no_dec --no_diff --n_train $n --seed $s --out ckpt/gnnn${n}_s$s.pt > logs_rev/gnnn${n}_s$s.txt 2>&1"
done; done
} > logs_rev/jobs_train.txt
xargs -P 10 -I{} bash -c "{}" < logs_rev/jobs_train.txt
echo PHASE1_DONE > logs_rev/phase1.done

# ---------------- phase 2: retrieval evaluation (8 concurrent) ----------------
R="$PY code/run_v3.py"
{
for ds in musique 2wikimultihopqa hotpotqa; do
  for s in 0 1 2 3 4; do
    echo "$R $ds unicor3_s$s u3nr_s$s gnn_s$s >> logs_rev/eval_$ds.txt 2>&1"
    echo "FILL=800 TAG=_fill $R $ds unicor3_s$s gnn_s$s >> logs_rev/eval_$ds.txt 2>&1"
  done
  for s in 0 1 2; do
    echo "$R $ds u3nodiff_s$s u3noctrl_s$s u3noprov_s$s u3tf_s$s u3noconf_s$s u3soft_s$s pointer_s$s >> logs_rev/eval_$ds.txt 2>&1"
  done
  for n in 300 1000; do for s in 0 1; do echo "$R $ds u3n${n}_s$s gnnn${n}_s$s >> logs_rev/eval_$ds.txt 2>&1"; done; done
  echo "$R $ds dense_ft >> logs_rev/eval_$ds.txt 2>&1"
  for t in 0.3 0.5 0.7 0.9; do echo "STOP=suff:$t TAG=_suff$t $R $ds unicor3_s0 >> logs_rev/stop_$ds.txt 2>&1"; done
  for k in 2 3 4 5; do echo "STOP=fixed:$k TAG=_fix$k $R $ds unicor3_s0 >> logs_rev/stop_$ds.txt 2>&1"; done
  for t in 0.1 0.2 0.3; do echo "STOP=thresh:$t TAG=_thr$t $R $ds unicor3_s0 >> logs_rev/stop_$ds.txt 2>&1"; done
  echo "INJECT=1 TAG=_inj $R $ds unicor3_s0 u3nr_s0 u3noconf_s0 >> logs_rev/inject_$ds.txt 2>&1"
  for np_ in 20 60; do echo "NP=$np_ TAG=_np$np_ $R $ds unicor3_s0 >> logs_rev/np_$ds.txt 2>&1"; done
done
} > logs_rev/jobs_eval.txt
xargs -P 8 -I{} bash -c "{}" < logs_rev/jobs_eval.txt
echo PHASE2_DONE > logs_rev/phase2.done

# ---------------- phase 3: QA reading (2 vLLM readers concurrently) ----------------
RD="GMU=0.42 $PY code/reader.py"
BASE="bm25 dense kg_only fixed_fusion rrf hipporag hipporag2"
{
for ds in musique 2wikimultihopqa hotpotqa; do
  echo "$RD $ds $BASE dense_ft > logs_rev/qa_base_$ds.txt 2>&1"
  M=""; for s in 0 1 2 3 4; do M="$M unicor3_s$s unicor3_s${s}_fill u3nr_s$s gnn_s${s}_fill"; done
  echo "$RD $ds $M > logs_rev/qa_main_$ds.txt 2>&1"
  echo "$RD $ds u3nodiff_s0 u3noctrl_s0 u3noprov_s0 u3tf_s0 u3noconf_s0 u3soft_s0 pointer_s0 unicor3_s0_suff0.3 unicor3_s0_suff0.7 unicor3_s0_suff0.9 unicor3_s0_fix2 unicor3_s0_fix3 unicor3_s0_fix4 unicor3_s0_fix5 unicor3_s0_thr0.1 unicor3_s0_thr0.2 unicor3_s0_thr0.3 > logs_rev/qa_abl_$ds.txt 2>&1"
done
} > logs_rev/jobs_qa.txt
xargs -P 2 -I{} bash -c "{}" < logs_rev/jobs_qa.txt
# reader robustness: Llama-3.1-8B int4 and Qwen2.5-7B fp16
for ds in musique 2wikimultihopqa hotpotqa; do
  for d in results_llama results_fp16; do mkdir -p $d/$ds; for m in dense hipporag2 dense_ft gnn_s0_fill unicor3_s0 unicor3_s0_fill; do cp results/$ds/$m.json $d/$ds/; done; done
done
{
for ds in musique 2wikimultihopqa hotpotqa; do
  echo "RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct $RD $ds dense hipporag2 dense_ft gnn_s0_fill unicor3_s0 unicor3_s0_fill > logs_rev/qa_llama_$ds.txt 2>&1"
done
} > logs_rev/jobs_qa2.txt
xargs -P 2 -I{} bash -c "{}" < logs_rev/jobs_qa2.txt
for ds in musique 2wikimultihopqa hotpotqa; do
  RESDIR=results_fp16 FP16=1 GMU=0.85 $PY code/reader.py $ds dense hipporag2 unicor3_s0 unicor3_s0_fill > logs_rev/qa_fp16_$ds.txt 2>&1
done
echo PHASE3_DONE > logs_rev/phase3.done
# ---------------- phase 4: graph audit ----------------
cp results/musique/fixed_fusion.json /dev/null
for ds in musique 2wikimultihopqa hotpotqa; do cp results/$ds/unicor3_s0.json results/$ds/unicor3.json; done
$PY code/judge.py > logs_rev/judge.txt 2>&1
echo ALL_DONE > logs_rev/all.done
