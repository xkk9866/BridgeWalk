#!/bin/bash
# BridgeWalk experiments, GPU 3 only (shared server: every stage waits for enough free memory on GPU 3).
# Track A (this script): seed-0 inference -> QA -> ablations (3 parallel lanes) -> sensitivity -> Llama reader.
# Track B (q_bridge2_s1.sh): seed-1 training -> inference -> QA.
cd ~/research/grag; source waitgpu.sh
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs_bridge; DS="musique 2wikimultihopqa hotpotqa"
qa() { waitgpu 12500; GMU=0.45 flock .reader.lock $PY code/reader.py "$@"; }

while [ ! -f ckpt/bridge_s0/heads.pt ]; do sleep 120; done
# main inference: one lane per dataset
for ds in $DS mhrag; do (waitgpu 3500; $PY code/bridgewalk.py infer $ds ckpt/bridge_s0 bridge_full full > $L/inf_full_$ds.txt 2>&1) & sleep 90; done; wait
for ds in $DS mhrag; do qa $ds bridge_full_SB300 bridge_full_SB450 bridge_full_SB600; done > $L/qa_full.txt 2>&1
echo ok > $L/qa_full.done
# ablations: one lane per dataset, variants in sequence
for ds in $DS; do (for v in nograph nostop qbridge allment oie; do waitgpu 3500; $PY code/bridgewalk.py infer $ds ckpt/bridge_s0 bridge_$v $v; done > $L/inf_abl_$ds.txt 2>&1) & sleep 90; done; wait
for ds in $DS; do M=""; for v in nograph nostop qbridge allment oie; do M="$M bridge_${v}_SB450"; done; qa $ds $M; done > $L/qa_abl.txt 2>&1
echo ok > $L/qa_abl.done
# sensitivity (MuSiQue, retrieval only) and dense first-hop candidates
(waitgpu 3500; KB=1 $PY code/bridgewalk.py infer musique ckpt/bridge_s0 bridge_K1 full; KB=3 $PY code/bridgewalk.py infer musique ckpt/bridge_s0 bridge_K3 full) > $L/sens_K.txt 2>&1 &
sleep 90; (waitgpu 3500; RB=4 $PY code/bridgewalk.py infer musique ckpt/bridge_s0 bridge_R4 full; RB=16 $PY code/bridgewalk.py infer musique ckpt/bridge_s0 bridge_R16 full) > $L/sens_R.txt 2>&1 &
sleep 90; (for ds in $DS; do waitgpu 3500; C0=dense $PY code/bridgewalk.py infer $ds ckpt/bridge_s0 bridge_dense0 full; done) > $L/inf_dense0.txt 2>&1 &
wait
for ds in $DS; do qa $ds bridge_dense0_SB450; done > $L/qa_dense0.txt 2>&1
# second reader
export RESDIR=results_llama READER=/home/yaokang/sra/models/Meta-Llama-3.1-8B-Instruct
for ds in $DS; do mkdir -p results_llama/$ds; cp results/$ds/bridge_full_SB450.json results/$ds/walk_full_SB450.json results_llama/$ds/; qa $ds bridge_full_SB450; done > $L/llama.txt 2>&1
echo ok > $L/all_A.done
