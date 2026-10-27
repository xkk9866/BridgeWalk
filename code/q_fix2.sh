#!/bin/bash
# Faster re-run on GPUs 0-3: matched-model walks spread over the idle GPUs 2/3, news walks on GPU 1, control QA on GPU 0.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs_ctrl; DS="musique 2wikimultihopqa hotpotqa"; B3="SB300 SB450 SB600"
rm -f results/*/fixtr.walks*.json results/mhrag/onpol.walks*.json
g=2; for ds in $DS; do for s in 0 1 2 3; do VAR=fixnb CUDA_VISIBLE_DEVICES=$g SHARD=$s NSHARD=4 $PY code/coverwalk.py infer $ds ckpt/fixnb_s0 fixtr local > $L/inf_fixtr_${ds}_$s.txt 2>&1 & g=$((5-g)); sleep 10; done; done
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=1 SHARD=$s NSHARD=3 $PY code/coverwalk.py infer mhrag ckpt/onpol_s0 onpol local > $L/inf_onpol_mhrag_$s.txt 2>&1 & sleep 10; done
qa() { g=$1; shift; CUDA_VISIBLE_DEVICES=$g GMU=0.55 $PY code/reader.py "$@"; }
for ds in $DS; do M="ctl_allment_marg_SB450"; for m in ctl_fixnb ctl_d3; do for b in $B3; do M="$M ${m}_marg_$b"; done; done; qa 0 $ds $M; done > $L/qa_ctl.txt 2>&1
wait
for ds in $DS; do $PY code/coverwalk.py pack $ds fixtr marg & done; $PY code/coverwalk.py pack mhrag onpol marg & wait
(qa 2 musique fixtr_marg_SB300 fixtr_marg_SB450 fixtr_marg_SB600) > $L/qa_f2.txt 2>&1 &
(qa 3 2wikimultihopqa fixtr_marg_SB300 fixtr_marg_SB450 fixtr_marg_SB600) > $L/qa_f3.txt 2>&1 &
(qa 0 hotpotqa fixtr_marg_SB300 fixtr_marg_SB450 fixtr_marg_SB600; qa 0 mhrag onpol_marg_SB300 onpol_marg_SB450 onpol_marg_SB600) > $L/qa_f0.txt 2>&1 &
wait
echo ok > $L/fix2.done
