REPO_ROOT="$(git rev-parse --show-toplevel)"
export TORCHINDUCTOR_ANNOTATE_TRAINING=1
export PYTHONPATH=${REPO_ROOT}/benchmarks
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 
export TORCHINDUCTOR_BENCHMARK_KERNEL=1 

python -u ${REPO_ROOT}/benchmarks/dynamo/timm_models.py \
--backend inductor \
--amp \
--performance \
--dashboard \
--only mixnet_l \
--disable-cudagraphs \
--training