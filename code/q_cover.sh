#!/bin/bash
# CoverWalk experiments, server GPU 4 only.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs_cover; mkdir -p $L ckpt/cw_mine; DS="musique 2wikimultihopqa hotpotqa"
inf() { ds=$1; md=$2; nm=$3; sc=$4; n=$5; for s in $(seq 0 $((n-1))); do SHARD=$s NSHARD=$n $PY code/coverwalk.py infer $ds $md $nm $sc > $L/inf_${nm}_${ds}_$s.txt 2>&1 & sleep 20; done; }
qa() { GMU=${GMU:-0.85} $PY code/reader.py "$@"; }

# 1. walks of the locally normalised BridgeWalk network: training questions (for learning) and test questions
for ds in hotpotqa 2wikimultihopqa musique; do for s in 0 1; do $PY code/coverwalk.py mine $ds ckpt/bridge2_s0 $s 2 > $L/mine_${ds}_$s.txt 2>&1 & sleep 20; done; done
for ds in $DS; do inf $ds ckpt/bridge2_s0 bwl local 1; done
wait
echo ok > $L/mine.done
# 2. globally normalised fine-tuning, two seeds in parallel; meanwhile pack the local walks
(CH=16 $PY code/coverwalk.py train ckpt/bridge2_s0 ckpt/cover_s0 0 > $L/train_s0.txt 2>&1) &
sleep 60
(CH=16 $PY code/coverwalk.py train ckpt/bridge2_s0 ckpt/cover_s1 1 > $L/train_s1.txt 2>&1) &
for ds in $DS; do $PY code/coverwalk.py pack $ds bwl mbr; $PY code/coverwalk.py pack $ds bwl marg; done > $L/pack_bwl.txt 2>&1
wait
echo ok > $L/train.done
# 3. test walks of CoverWalk (global energy)
for ds in $DS; do inf $ds ckpt/cover_s0 cw global 2; done
inf mhrag ckpt/cover_s0 cw global 2
wait
for ds in $DS; do inf $ds ckpt/cover_s1 cws1 global 2; inf $ds ckpt/cover_s0 cwloc local 1; done
for ds in $DS mhrag; do $PY code/coverwalk.py pack $ds cw mbr; $PY code/coverwalk.py pack $ds cw marg; done > $L/pack_cw.txt 2>&1
wait
for ds in $DS; do $PY code/coverwalk.py pack $ds cws1 mbr; $PY code/coverwalk.py pack $ds cwloc mbr; done > $L/pack_cws1.txt 2>&1
echo ok > $L/inf.done
# 4. reading (Qwen2.5-7B AWQ), then the second reader
for ds in $DS; do qa $ds cw_SB300 cw_SB450 cw_SB600 cw_SB800 cw_marg_SB450 bwl_SB450 bwl_marg_SB450 cws1_SB300 cws1_SB450 cws1_SB600 cwloc_SB450; done > $L/qa.txt 2>&1
qa mhrag cw_SB300 cw_SB450 cw_SB600 >> $L/qa.txt 2>&1
echo ok > $L/qa.done
export RESDIR=results_llama2 READER=/home/yaokang/research/testimony_rag/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
for ds in $DS; do cp results/$ds/cw_SB450.json results_llama2/$ds/; qa $ds cw_SB450; done > $L/llama.txt 2>&1
echo ok > $L/all.done
