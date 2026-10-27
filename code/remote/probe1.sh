cd ~/research/grag
mkdir -p logs_thread
export HF_HUB_OFFLINE=1
for ds in musique 2wikimultihopqa hotpotqa; do
  nohup setsid .venv/bin/python code/probe_thread.py $ds bw bwth > logs_thread/probe_$ds.txt 2>&1 < /dev/null &
done
echo launched
