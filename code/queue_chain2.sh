#!/bin/bash
# Graph-vs-text proposer ablation for ChainCoR (GPU 6 after the main chain queue finishes its QA lane on GPU 6).
cd ~/research/grag
export HF_HUB_OFFLINE=1 PY=.venv/bin/python PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=6
while pgrep -f "code/chain_infe[r]" > /dev/null; do sleep 60; done
for ds in musique 2wikimultihopqa hotpotqa; do
  PROPOSER=dense CANDSUF=_dense $PY code/chain_data.py $ds 12 >> logs_chain/dense_prop.txt 2>&1
  CANDSUF=_dense $PY code/chain_infer.py $ds ckpt/chain_pair_s0 pair chainDP_s0 >> logs_chain/dense_prop.txt 2>&1
done
while [ ! -f logs_chain/all.done ]; do sleep 60; done
export GMU=0.85
for ds in musique 2wikimultihopqa hotpotqa; do $PY code/reader.py $ds chainDP_s0_B450 chainDP_s0_B600 > logs_chain/qa_dense_prop_$ds.txt 2>&1; done
echo done > logs_chain/dense.done
