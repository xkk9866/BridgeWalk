cd ~/research/grag
export HF_HUB_OFFLINE=1 GMU=0.55
i=0
for ds in musique 2wikimultihopqa hotpotqa; do
  g=$((4+i)); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g nohup setsid .venv/bin/python code/reader.py $ds bw_SB300 bw_SB450 bw_SB600 bwth_SB300 bwth_SB450 bwth_SB600 > logs_thread/qa1_$ds.txt 2>&1 < /dev/null &
done
echo ok
