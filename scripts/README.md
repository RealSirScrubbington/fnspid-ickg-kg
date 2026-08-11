# Cluster launch scripts

Records of the exact GPU-cluster invocations used for the thesis runs,
**preserved verbatim**: they hardcode cluster paths, source
`goldbug_env.sh` from the repo root (their original location), and call
sibling scripts by bare name, so they document what ran rather than serve
as turnkey tools. The runnable equivalents are the plain Python commands
in [REPRODUCING.md](../REPRODUCING.md) — nothing here contains logic of
its own beyond sharding, environment activation, and resumable `nohup`
chaining.

| script | purpose |
|---|---|
| `run_build.sh` | data-parallel ICKG extraction: one vLLM instance per GPU over interleaved shards (Stage 2) |
| `setup_goldbug.sh`, `goldbug_env.sh`, `setup_remote.sh`, `download_goldbug.sh` | UCL cluster environment bootstrap |
| `download_model.py`, `verify_env.py` | model prefetch and environment sanity check |
| `run_judge.sh`, `run_judge2.sh`, `run_judge_arms.sh`, `run_judge_r15.sh` | 14B audit-judge passes over the audit packs, per detector arm (Stage 5) |
| `run_finetune.sh` | 1.5B judge LoRA fine-tune (Stage 6) |
| `run_topk.sh`, `launch_topk.sh` | replica RE-GCN run dumping top-10 candidates per test query (error anatomy) |
| `run_ablation_prowl.sh` | velocity-feature ablation arms, 3 seeds (Stage 8) |
| `run_night.sh`, `run_night2.sh`, `launch_night.sh` | chained overnight batches: judge deployment sweep + template-filtered RE-GCN retrain, with per-stage DONE flags |
