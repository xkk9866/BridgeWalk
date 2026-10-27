#!/bin/bash
cd ~/research/grag/data/raw_large
timeout 3000 curl -sL -o mirror-train-00001.parquet https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/train-00001-of-00002.parquet
ls -la > dl3.txt
