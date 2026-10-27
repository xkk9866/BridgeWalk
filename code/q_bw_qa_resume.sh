#!/bin/bash
# Resume the BridgeWalk QA runs of q_bridge4.sh that died on a vLLM memory-profiling error (GPU 3 only).
cd ~/research/grag
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false CH=4
PY=.venv/bin/python; L=logs_bridge
qa() { for t in 1 2 3; do GMU=0.45 flock .reader.lock $PY code/reader.py "$@" && return; sleep 120; done; }
for ds in musique hotpotqa; do qa $ds bw_SB300 bw_SB450 bw_SB600 bw_SB800 bwth_SB300 bwth_SB450 bwth_SB600; done > $L/qa2_resume.txt 2>&1
qa 2wikimultihopqa bw_SB800 >> $L/qa2_resume.txt 2>&1
qa mhrag bw_SB300 bw_SB450 bw_SB600 >> $L/qa2_resume.txt 2>&1
echo ok > $L/qa2_resume.done
