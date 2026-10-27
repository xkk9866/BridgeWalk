#!/bin/bash
# usage: repack_lanes.sh "<ds list>" "<group1>" "<group2>" ...   (one detached process per dataset x group)
cd ~/research/grag; DSL=$1; shift
for ds in $DSL; do for g in "$@"; do
  nohup setsid env EMB_DEV=cpu CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python code/repack.py $ds $g > "logs_bridge/repack_${ds}_${g%% *}.txt" 2>&1 < /dev/null &
done; done
