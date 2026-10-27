cd ~/research/grag; export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PY=.venv/bin/python
$PY code/train.py --hop --out ckpt/unicor_hop.pt > logs_train_hop.txt 2>&1 || { tail -20 logs_train_hop.txt; exit 1; }
for ds in musique 2wikimultihopqa hotpotqa; do WIDE=1 $PY code/run_retrieval.py $ds unicor_hop; done
echo HOP_DONE
