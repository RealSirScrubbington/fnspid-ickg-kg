#!/usr/bin/env bash
cd /cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
hostname
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
chmod +x run_topk.sh
nohup bash run_topk.sh > topk_nohup.log 2>&1 < /dev/null &
echo "topk launched pid $!"
