#!/bin/bash
# Re-run of the pieces that failed: QA of the controls (GPU 0), matched-model walks (GPU 1), news walks of the DAgger model.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=logs_ctrl; DS="musique 2wikimultihopqa hotpotqa"
B3="SB300 SB450 SB600"
for ds in $DS; do for s in 0 1; do rm -f results/$ds/fixtr.walks$s.json; VAR=fixnb CUDA_VISIBLE_DEVICES=1 SHARD=$s NSHARD=2 $PY code/coverwalk.py infer $ds ckpt/fixnb_s0 fixtr local > $L/inf_fixtr_${ds}_$s.txt 2>&1 & sleep 20; done; done
for s in 0 1; do CUDA_VISIBLE_DEVICES=0 SHARD=$s NSHARD=2 $PY code/coverwalk.py infer mhrag ckpt/onpol_s0 onpol local > $L/inf_onpol_mhrag_$s.txt 2>&1 & sleep 20; done
qa() { g=$1; shift; CUDA_VISIBLE_DEVICES=$g GMU=0.6 $PY code/reader.py "$@"; }
for ds in $DS; do M="ctl_allment_marg_SB450"; for m in ctl_fixnb ctl_d3; do for b in $B3; do M="$M ${m}_marg_$b"; done; done; qa 0 $ds $M; done > $L/qa_ctl.txt 2>&1
wait
for ds in $DS; do $PY code/coverwalk.py pack $ds fixtr marg; done > $L/pack_fixtr.txt 2>&1
$PY code/coverwalk.py pack mhrag onpol marg >> $L/pack_fixtr.txt 2>&1
(for ds in musique 2wikimultihopqa; do qa 0 $ds fixtr_marg_SB300 fixtr_marg_SB450 fixtr_marg_SB600; done) > $L/qa_fixtr0.txt 2>&1 &
(qa 1 hotpotqa fixtr_marg_SB300 fixtr_marg_SB450 fixtr_marg_SB600; qa 1 mhrag onpol_marg_SB300 onpol_marg_SB450 onpol_marg_SB600) > $L/qa_fixtr1.txt 2>&1 &
wait
echo ok > $L/fix.done
