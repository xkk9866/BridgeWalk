cd ~/research/grag; export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PY=.venv/bin/python
for ds in musique 2wikimultihopqa hotpotqa; do FILL=900 TAG=_fill $PY code/run_retrieval.py $ds unicor; done
for ds in musique 2wikimultihopqa hotpotqa; do $PY code/reader.py $ds unicor_fill; done
echo FILL_DONE
