#!/usr/bin/env bash
set -euo pipefail

# Run small, focused Inductor tests by component.
# Usage examples:
#   ./codex/scripts/inductor_component_tests.sh dynamo
#   ./codex/scripts/inductor_component_tests.sh codegen
#   DRY_RUN=1 ./codex/scripts/inductor_component_tests.sh cache

DRY_RUN=${DRY_RUN:-}
run() {
  echo "+ $*"
  if [[ -z "${DRY_RUN}" ]]; then
    eval "$@"
  fi
}

component=${1:-}
if [[ -z "$component" ]]; then
  echo "Specify a component: dynamo|functionalize|postgrad|ir|codegen|runtime|cache"
  exit 1
fi

case "$component" in
  dynamo)
    run "pytest -q test/inductor/test_control_flow.py::test_cond_simple_control_flow"
    run "pytest -q test/inductor/test_torchinductor_dynamic_shapes.py::test_arange_dynamic"
    ;;
  functionalize)
    run "pytest -q test/inductor/test_auto_functionalize.py::test_auto_functionalize_can_with_default"
    run "pytest -q test/inductor/test_decompose_mem_bound_mm.py::test_decompose_bmm"
    ;;
  postgrad)
    run "TORCH_LOGS=post_grad_graphs pytest -q test/inductor/test_custom_post_grad_passes.py::test_custom_joint_pass_post"
    run "pytest -q test/inductor/test_split_cat_fx_passes.py::test_split_normalization"
    ;;
  ir)
    run "pytest -q test/inductor/test_loop_ordering.py::TestTiling"
    run "pytest -q test/inductor/test_segmented_tree.py::test_basic_construction"
    ;;
  codegen)
    run "pytest -q test/inductor/test_triton_kernels.py::test_triton_kernel_with_kernel_param"
    run "pytest -q test/inductor/test_max_autotune.py::TestMaxAutotune::test_max_autotune_decompose_k_dynamic_input"
    ;;
  runtime)
    run "pytest -q test/inductor/test_cudagraph_trees.py::test_run_simple"
    run "pytest -q test/inductor/test_static_cuda_launcher.py::test_basic"
    ;;
  cache)
    run "pytest -q test/inductor/test_codecache.py::TestFxGraphCache::test_cache_guard"
    run "pytest -q test/inductor/test_cudacodecache.py::test_cuda_load"
    ;;
  *)
    echo "Unknown component: $component"
    exit 1
    ;;
esac

