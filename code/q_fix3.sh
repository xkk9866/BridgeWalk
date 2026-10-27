#!/bin/bash
# Follow-up: control QA on GPU 2 now; when the running matched-model / news walks finish, pack and read them on GPUs 2/3.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 TOKENIZERS_PARALLELISM=false
L=logs_ctrl; DS="musique 2wikimultihopqa hotpotqa"; B3="SB300 SB450 SB600"
qa() { g=$1; shift; CUDA_VISIBLE_DEVICES=$g GMU=0.85 $PY code/reader.py "$@"; }
(for ds in musique 2wikimultihopqa; do M="ctl_allment_marg_SB450"; for m in ctl_fixnb ctl_d3; do for b in $B3; do M="$M ${m}_marg_$b"; done; done; qa 2 $ds $M; done) > $L/qa_ctl2.txt 2>&1 &
(M="ctl_allment_marg_SB450"; for m in ctl_fixnb ctl_d3; do for b in $B3; do M="$M ${m}_marg_$b"; done; done; qa 3 hotpotqa $M) > $L/qa_ctl3.txt 2>&1 &
while pgrep -f "coverwalk.py infer" > /dev/null; do sleep 30; done
for ds in $DS; do $PY code/coverwalk.py pack $ds fixtr marg > $L/pack_fixtr_$ds.txt 2>&1 & done
$PY code/coverwalk.py pack mhrag onpol marg > $L/pack_onpol_mhrag.txt 2>&1 &
wait
(qa 2 musique fixtr_marg_SB300 fixtr_marg_SB450 fixtr_marg_SB600; qa 2 hotpotqa fixtr_marg_SB300 fixtr_marg_SB450 fixtr_marg_SB600) > $L/qa_f2.txt 2>&1 &
(qa 3 2wikimultihopqa fixtr_marg_SB300 fixtr_marg_SB450 fixtr_marg_SB600; qa 3 mhrag onpol_marg_SB300 onpol_marg_SB450 onpol_marg_SB600) > $L/qa_f3.txt 2>&1 &
wait
echo ok > $L/fix3.done
