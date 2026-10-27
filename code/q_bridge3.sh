#!/bin/bash
# BridgeWalk with the sufficiency-trained halting read-out (bridge2_*), GPU 3 only.
cd ~/research/grag; source waitgpu.sh
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 SUBJ=1 CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CH=4
L=logs_bridge; DS="musique 2wikimultihopqa hotpotqa"
qa() { for t in 1 2 3; do waitgpu 12500; GMU=0.45 flock .reader.lock $PY code/reader.py "$@" && return; sleep 300; done; }
rp() { ds=$1; shift; EMB_DEV=cpu $PY code/repack.py $ds "$@"; }   # SB1000 / SB2000 / top5 from the SB450 ranking
inf() { for t in 1 2 3 4; do waitgpu 3000; $PY code/bridgewalk.py infer "$@" && return; sleep 300; done; }      # retry: shared GPU may OOM
trn() { for t in 1 2 3 4; do waitgpu 2200; $PY code/bridgewalk.py train ckpt/joint_s0 ckpt/$1 $2 && return; sleep 300; done; }

# 1. two seeds of the corrected model, in parallel
(trn bridge2_s0 0 > $L/train2_s0.txt 2>&1) &
sleep 120
(trn bridge2_s1 1 > $L/train2_s1.txt 2>&1) &
# meanwhile: QA of the old model (halting trained on gold passages only) once its inference is complete
while pgrep -f "bridgewalk.py infer (hotpotqa|mhrag) ckpt/bridge_s0" > /dev/null; do sleep 120; done
for ds in $DS; do rp $ds bridge_full; done > $L/rp_old.txt 2>&1
for ds in $DS; do qa $ds bridge_full_SB300 bridge_full_SB450 bridge_full_SB600 bridge_full_SB1000; done > $L/qa_old.txt 2>&1
# baselines at the standard context sizes (rankings re-packed by code/repack.py)
for ds in $DS; do M=""; for m in dense mdrT hybrid hipporag2 rerankZS point gnn chainS_s0 walk_full; do M="$M ${m}_SB1000 ${m}_SB2000 ${m}_top5"; done; qa $ds $M; done > $L/qa_base_large.txt 2>&1
wait
echo ok > $L/train2.done

# 2. main inference, seed 0 and seed 1, one lane per dataset
for ds in $DS mhrag; do (inf $ds ckpt/bridge2_s0 bw full > $L/inf2_$ds.txt 2>&1; rp $ds bw >> $L/inf2_$ds.txt 2>&1) & sleep 60; done
for ds in $DS; do (inf $ds ckpt/bridge2_s1 bws1 full > $L/inf2s1_$ds.txt 2>&1; rp $ds bws1 >> $L/inf2s1_$ds.txt 2>&1) & sleep 60; done
wait
# seed 2 trains while QA runs
(trn bridge2_s2 2 > $L/train2_s2.txt 2>&1;
 for ds in $DS; do inf $ds ckpt/bridge2_s2 bws2 full; rp $ds bws2; done > $L/inf2s2.txt 2>&1) &
for ds in $DS; do qa $ds bw_SB300 bw_SB450 bw_SB600 bw_SB1000 bw_SB2000 bw_top5; done > $L/qa2.txt 2>&1
qa mhrag bw_SB300 bw_SB450 bw_SB600 >> $L/qa2.txt 2>&1
for ds in $DS; do qa $ds bws1_SB300 bws1_SB450 bws1_SB600 bws1_SB1000; done > $L/qa2s1.txt 2>&1
echo ok > $L/qa2.done

# 3. ablations (seed 0), one lane per dataset
for ds in $DS; do (for v in nograph nostop qbridge allment oie; do inf $ds ckpt/bridge2_s0 bw_$v $v; done > $L/inf2_abl_$ds.txt 2>&1) & sleep 60; done
wait
(KB=1 inf musique ckpt/bridge2_s0 bw_K1 full; KB=3 inf musique ckpt/bridge2_s0 bw_K3 full; RB=4 inf musique ckpt/bridge2_s0 bw_R4 full; RB=16 inf musique ckpt/bridge2_s0 bw_R16 full) > $L/sens2.txt 2>&1 &
(for ds in $DS; do C0=dense inf $ds ckpt/bridge2_s0 bw_dense0 full; done) > $L/inf2_dense0.txt 2>&1 &
for ds in $DS; do M=""; for v in nograph nostop qbridge allment oie; do M="$M bw_${v}_SB450"; done; qa $ds $M; done > $L/qa2_abl.txt 2>&1
wait
for ds in $DS; do qa $ds bw_dense0_SB450; done >> $L/qa2_abl.txt 2>&1
while [ ! -f ckpt/bridge2_s2/heads.pt ] || pgrep -f "bridge2_s2" > /dev/null; do sleep 300; done
for ds in $DS; do qa $ds bws2_SB300 bws2_SB450 bws2_SB600 bws2_SB1000; done > $L/qa2s2.txt 2>&1
echo ok > $L/abl2.done

# 4. stronger and different readers on the shared rankings (Qwen2.5-14B AWQ, Llama-3.1-8B AWQ)
KEY="dense hybrid hipporag2 rerankZS point chainS_s0 walk_full bw"
rd() { R=$1; DIR=$2; NEED=$3; G=$4; shift 4
  for ds in $DS; do mkdir -p $DIR/$ds; M=""
    for m in $KEY; do for t in SB450 SB1000 top5; do [ -f results/$ds/${m}_$t.json ] && cp results/$ds/${m}_$t.json $DIR/$ds/ && M="$M ${m}_$t"; done; done
    waitgpu $NEED; RESDIR=$DIR READER=$R GMU=$G flock .reader.lock $PY code/reader.py $ds $M; done; }
rd /home/yaokang/research/testimony_rag/models/Qwen2.5-14B-Instruct-AWQ results_q14 15000 0.58 > $L/qa_q14.txt 2>&1
rd /home/yaokang/research/testimony_rag/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 results_llama2 12500 0.45 > $L/qa_llama2.txt 2>&1
echo ok > $L/all3.done
