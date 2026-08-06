#!/bin/bash
# One-shot launcher for the chained night run. Run on the GPU host:
#   bash /cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg/launch_night.sh
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
hostname
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
rm -f NIGHT_DONE JUDGE_SCORES_DONE REGCN_TF_DONE
nohup bash run_night.sh > night_nohup.log 2>&1 &
echo "night run launched pid $!"
