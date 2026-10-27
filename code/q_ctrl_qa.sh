#!/bin/bash
# Packing (BridgeWalk's marginal packer) and Qwen2.5-7B-AWQ reading for the reviewer-round runs; GPUs 0 and 1.
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 TOKENIZERS_PARALLELISM=false
L=logs_ctrl; DS="musique 2wikimultihopqa hotpotqa"
while [ ! -f $L/stage2.done ]; do sleep 60; done
pk() { $PY code/coverwalk.py pack $1 $2 marg; }
(for ds in $DS; do pk $ds ctl_fixnb; pk $ds ctl_d3; pk $ds ctl_allment; pk open_$ds bw; done
 true) > $L/pack1.txt 2>&1
while [ ! -f $L/stage2.done ]; do sleep 60; done
(for ds in $DS mhrag; do pk $ds onpol; done; for ds in $DS; do pk $ds fixtr; done) > $L/pack2.txt 2>&1
qa() { g=$1; shift; CUDA_VISIBLE_DEVICES=$g GMU=0.85 $PY code/reader.py "$@"; }
B3="SB300 SB450 SB600"
(for ds in $DS; do M=""; for m in onpol fixtr ctl_fixnb ctl_d3; do for b in $B3; do M="$M ${m}_marg_$b"; done; done; qa 0 $ds $M; done
 qa 0 mhrag onpol_marg_SB300 onpol_marg_SB450 onpol_marg_SB600) > $L/qa0.txt 2>&1 &
(for ds in $DS; do qa 1 $ds ctl_allment_marg_SB450 onpol_marg_SB800 fixtr_marg_SB800; done

 for ds in $DS; do qa 1 open_$ds bw_marg_SB450 bw_marg_SB600; done) > $L/qa1.txt 2>&1 &
wait
echo ok > $L/qa.done
