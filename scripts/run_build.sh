#!/usr/bin/env bash
# Stage 2 data-parallel ICKG extraction.
#   usage: run_build.sh [subset.parquet] [output-prefix]
#   - one vLLM instance per GPU (interleaved shards 0/2, 1/2)
#   - CSV-append checkpointing (resumable: re-run to continue from where it died)
#   - cleans vLLM GPU-zombies and verifies both GPUs are free before launching
set -u
PROJ=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$PROJ"; source goldbug_env.sh; . .venv-blackwell/bin/activate
INPUT=${1:-data/subset_200k.parquet}
PREFIX=${2:-data/tripl_full}
CHUNK=1000; MAXTOK=1280

echo "[build] cleaning vLLM zombies"
pkill -u dmaruev -9 -f "extract.py"  2>/dev/null
pkill -u dmaruev -9 -f "EngineCore"  2>/dev/null
sleep 6
for g in 0 1; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')
  echo "[build] GPU$g used=${used} MiB"
  if [ "$used" -ge 2000 ]; then echo "[build] GPU$g NOT free -- aborting"; exit 1; fi
done

echo "[build] $INPUT -> ${PREFIX}_{0,1}.csv  (chunk=$CHUNK, max_new=$MAXTOK)"
CUDA_VISIBLE_DEVICES=0 python extract.py --input "$INPUT" --out ${PREFIX}_0.csv \
  --shard 0/2 --tp 1 --chunk $CHUNK --max-new-tokens $MAXTOK --max-model-len 8192 > ${PREFIX}_g0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python extract.py --input "$INPUT" --out ${PREFIX}_1.csv \
  --shard 1/2 --tp 1 --chunk $CHUNK --max-new-tokens $MAXTOK --max-model-len 8192 > ${PREFIX}_g1.log 2>&1 &
P1=$!
echo "[build] g0 pid=$P0  g1 pid=$P1  (logs: ${PREFIX}_g0.log / ${PREFIX}_g1.log)"

rc=0
wait $P0; r0=$?; echo "[build] g0 finished rc=$r0"; [ $r0 -ne 0 ] && rc=1
wait $P1; r1=$?; echo "[build] g1 finished rc=$r1"; [ $r1 -ne 0 ] && rc=1
pkill -u dmaruev -9 -f "EngineCore" 2>/dev/null
echo "BUILD_DONE rc=$rc"
