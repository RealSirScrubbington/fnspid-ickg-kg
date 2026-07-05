# Source before any GPU run on goldbug. Redirects EVERY cache off the quota-full
# HOME (incl. flashinfer/vLLM JIT caches) and disables JIT-sampler + telemetry.
PROJ=/cs/student/projects3/csml/2025/dmaruev/fnspid-ickg-kg
export HOME="$PROJ/fakehome"
export HF_HOME="$PROJ/cache/hf"
export TRITON_CACHE_DIR="$PROJ/cache/triton"
export XDG_CACHE_HOME="$PROJ/cache/xdg"
export XDG_CONFIG_HOME="$PROJ/cache/xdgconfig"
export VLLM_CACHE_ROOT="$PROJ/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$PROJ/cache/inductor"
export FLASHINFER_WORKSPACE_BASE="$PROJ/cache/flashinfer"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING
mkdir -p "$HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
