#!/bin/bash
# Second reader (Llama-3.1-8B-Instruct AWQ int4) at B=450 on the shared rankings; waits for q_bw_qa_resume.sh.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false CH=4
PY=.venv/bin/python; L=logs_bridge; DIR=results_llama2
while [ ! -f $L/qa2_resume.done ]; do sleep 60; done
for ds in musique 2wikimultihopqa hotpotqa; do mkdir -p $DIR/$ds; M=""
  for m in dense mdrT hybrid hipporag2 rerankZS point gnn chainS_s0 walk_full bw; do
    [ -f results/$ds/${m}_SB450.json ] && cp results/$ds/${m}_SB450.json $DIR/$ds/ && M="$M ${m}_SB450"; done
  for t in 1 2 3; do RESDIR=$DIR READER=/home/yaokang/research/testimony_rag/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 GMU=0.45 \
    flock .reader.lock $PY code/reader.py $ds $M && break; sleep 120; done
done > $L/qa_llama_bw.txt 2>&1
echo ok > $L/qa_llama_bw.done
