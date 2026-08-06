#!/usr/bin/env bash
# Second-judge run for inter-judge agreement: Phi-4 (14B, MIT, non-Qwen family) judges the
# same frozen packs as the primary Qwen judge, themes mode, identical rubric and prompt.
# (First attempt used Mistral-7B, which labelled ~99% REAL and failed the control gate -
# archived as *_MISTRAL_FAILED_GATE.csv; a competent second judge needs 14B-class capacity.)
# Outputs: audit/theme_judge2_labels.csv, audit/theme_judge2_labels_tf.csv, JUDGE2_DONE
set -e
P=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
cd "$P"
source goldbug_env.sh 2>/dev/null || true
export TMPDIR="$P/cache/tmp"; mkdir -p "$TMPDIR"
rm -f audit/JUDGE2_DONE audit/judge2_failures.txt
M1="stelterlab/phi-4-AWQ"
cd "$P/audit"
run() {
  "$P/.venv-judge/bin/python" theme_judge.py --packs "$1" --out "$2" --mode themes --model "$M1" \
    || echo "FAILED $1" >> judge2_failures.txt
}
echo "=== judge2 lenient $(date) ==="
run audit_packs.jsonl theme_judge2_labels.csv
echo "=== judge2 tf $(date) ==="
run audit_packs_tf.jsonl theme_judge2_labels_tf.csv
touch JUDGE2_DONE
echo "=== all done $(date) ==="
