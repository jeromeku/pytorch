# TorchInductor Tests — Detailed Stage Mapping and Isolated Runners

Purpose: map test/inductor coverage to the major torch.compile → AOTAutograd → Inductor pipeline stages, with concrete example tests (path:line) and ready-to-run commands to exercise each component in isolation.

Updated: 2025-09-09

---

## How To Run One Test In Isolation
- Use pytest’s node ids to run a single test or class.
- Optional: set `TORCH_LOGS`, `PYTORCH_TEST_WITH_SLOW`, or Inductor/Dynamo config flags.

Examples
- Run one test function: `pytest -q test/inductor/test_codecache.py::TestFxGraphCache::test_cache_guard -q`
- Filter by pattern: `pytest -q test/inductor/test_max_autotune.py -k decompose_k_dynamic_input`
- Enable post-grad FX logs: `TORCH_LOGS=post_grad_graphs pytest -q test/inductor/test_torchinductor.py::TestCase::test_bool`

See `codex/scripts/inductor_component_tests.sh` for scripted shortcuts.

---

## Stage 2.1 — Dynamo Capture & Guards
Focus: graph breaks, guard precision, dynamic shapes, stance/counters, control flow.

Key modules (representative tests)
- test/inductor/test_control_flow.py
  - test_cond_simple_control_flow:324
  - test_cond_simple_with_int_closure:338
  - test_cond_control_flow_with_precomputed_size:366
- test/inductor/test_torchinductor.py
  - setUp/tearDown resets: 341, 342, 348
  - test_bool:989
- test/inductor/test_torchinductor_dynamic_shapes.py
  - test_constant_fold_uniform_value_dynamic:182
  - test_arange_dynamic:227
- test/inductor/test_torchinductor_codegen_dynamic_shapes.py
  - (module exercises codegen paths under dynamic shapes)

Commands
- Basic capture and guards: `pytest -q test/inductor/test_control_flow.py::test_cond_simple_control_flow`
- Dynamic shapes smoke: `pytest -q test/inductor/test_torchinductor_dynamic_shapes.py::test_arange_dynamic`

Notes
- Inspect FX post-grad after capture using `logs_to_string("torch._inductor.compile_fx", "post_grad_graphs")` in a local repro.

---

## Stage 2.2 — Functionalization & Decompositions
Focus: mutation → functional graphs, PrimTorch decompositions, custom decompositions.

Key modules
- test/inductor/test_auto_functionalize.py
  - test_auto_functionalize_can_with_default:20
  - test_auto_functionalize_can_with_none_return:43
- test/inductor/test_decompose_mem_bound_mm.py
  - test_decompose_bmm:105
  - test_decompose_bmm_cpu:142
- test/inductor/test_mmdecomp.py
  - (dense decomposition tests; e.g., op-specific paths)
- test/inductor/test_pad_mm.py
  - test_pad_mm_dyn_m:31
  - test_pad_mm_dyn_n:101
- test/inductor/test_binary_folding.py
  - test_conv_binary_folding:41
  - test_conv_bn_folding:158
- test/inductor/test_inplacing_pass.py
  - (reinplacing / mutation canonicalization)

Commands
- Decompose BMM: `pytest -q test/inductor/test_decompose_mem_bound_mm.py::test_decompose_bmm`
- Pad-MM canonicalization: `pytest -q test/inductor/test_pad_mm.py::test_pad_mm_dyn_m`

---

## Stage 2.3 — FX Post-Grad Shaping & Partitioning
Focus: post-grad normalization, custom passes, partitioners, pattern rewrites.

Key modules
- test/inductor/test_custom_post_grad_passes.py
  - test_custom_joint_pass_pre:147
  - test_custom_joint_pass_post:159
- test/inductor/test_custom_partitioner_fn.py
  - test_custom_partitioner_fn:30
- test/inductor/test_split_cat_fx_passes.py
  - test_split_normalization:36
  - test_consecutive_split_merge:146
- test/inductor/test_split_cat_fx_aten_passes.py
  - test_split_cat_post_grad:259
  - test_select_cat_post_grad:328
- test/inductor/test_pattern_matcher.py
  - test_mm_plus_mm:88
  - test_fused_int_mm_mul:151
- test/inductor/test_mkldnn_pattern_matcher.py
  - rich suite (e.g., pattern matcher variants)

Commands
- Verify a custom pass runs: `pytest -q test/inductor/test_custom_post_grad_passes.py::test_custom_joint_pass_post`
- Split/Cat normalization: `pytest -q test/inductor/test_split_cat_fx_passes.py::test_split_normalization`

---

## Stage 2.4 — Inductor IR & Fusion
Focus: scheduler legality, loop IR transforms, fusion groups, segmented trees.

Key modules
- test/inductor/test_inductor_scheduler.py
  - (scheduler legality and transformations)
- test/inductor/test_loop_ordering.py
  - class TestTiling:955 (tiling/ordering suite)
- test/inductor/test_kernel_optimization.py
  - class TestKernelOptimization:31
- test/inductor/test_segmented_tree.py
  - test_basic_construction:46
  - test_max_query_matches_naive:57
- test/inductor/test_group_batch_fusion.py
  - (grouped fusion checks for attention-like patterns)
- test/inductor/test_fx_fusion.py
  - (fx-driven fusion expectations)

Commands
- Loop ordering/tiling: `pytest -q test/inductor/test_loop_ordering.py::TestTiling`
- Segmented tree ops: `pytest -q test/inductor/test_segmented_tree.py::test_basic_construction`

---

## Stage 2.5 — Codegen & Autotune (Triton/CUTLASS)
Focus: Triton codegen, CUTLASS backend, autotune knobs, kernel templates.

Key modules
- test/inductor/test_triton_kernels.py
  - test_triton_kernel_with_kernel_param:100
  - test_triton_kernel_functionalize:167
- test/inductor/test_triton_heuristics.py
  - test_triton_config:82
  - test_artificial_zgrid:132
- test/inductor/test_cutlass_backend.py
  - test_check_paths:203
  - test_max_autotune_cutlass_threshold:218
- test/inductor/test_max_autotune.py
  - test_max_autotune_decompose_k_dynamic_input:1165
  - plus extensive suites for precompile/subproc/remote cache
- test/inductor/test_best_config.py
  - test_best_config_has_triton_cache_key:52
- test/inductor/test_codegen_triton.py
  - test_config_of_sizearg:35
- test/inductor/test_combo_kernels.py
  - test_reduce_functions:79
  - test_activation_functions:59

Commands
- Triton kernel param smoke: `pytest -q test/inductor/test_triton_kernels.py::test_triton_kernel_with_kernel_param`
- CUTLASS tuning threshold: `pytest -q test/inductor/test_cutlass_backend.py::test_max_autotune_cutlass_threshold`
- Autotune decompose-k: `pytest -q test/inductor/test_max_autotune.py::TestMaxAutotune::test_max_autotune_decompose_k_dynamic_input`

---

## Stage 2.6 — Runtime: CUDA Graphs & Collectives
Focus: cudagraph recording trees, replay semantics, streams, runtime buffer policies.

Key modules
- test/inductor/test_cudagraph_trees.py
  - test_run_simple:209
  - test_rng_trees:239
- test/inductor/test_static_cuda_launcher.py
  - test_basic:68
  - test_signed_integers:119
- test/inductor/test_memory_planning.py
  - (buffer planning & lifetimes)
- test/inductor/test_device_assert.py
  - test_fn:20
  - test_fn:40

Commands
- CUDAGraphs simple run: `pytest -q test/inductor/test_cudagraph_trees.py::test_run_simple`
- Static CUDA launcher basics: `pytest -q test/inductor/test_static_cuda_launcher.py::test_basic`

---

## Stage 2.7 — Caching & Hot-Load
Focus: FX graph cache, kernel caches, artifact (de)serialization, remote caches.

Key modules
- test/inductor/test_codecache.py
  - test_cache_load_function:172
  - test_cache_guard:1447
  - TestAutotuneCache, TestStandaloneCompile suites
- test/inductor/test_remote_cache.py
  - test_normal_logging:45
  - test_failure_logging:60
- test/inductor/test_cudacodecache.py
  - test_cuda_load:41
  - test_compilation_error:69
- test/inductor/test_compile_subprocess.py
  - test_progressive:100
  - test_async:178

Commands
- Cache guard divergence: `pytest -q test/inductor/test_codecache.py::TestFxGraphCache::test_cache_guard`
- CUDA code cache smoke: `pytest -q test/inductor/test_cudacodecache.py::test_cuda_load`

---

## Other Notable Families (Cross-Cutting)
- Opinfo and parity
  - test/inductor/test_torchinductor_opinfo.py::test_comprehensive:1153
  - test/inductor/test_op_completeness.py (module)
- Quantization and FP8
  - test/inductor/test_quantization.py::test_activation_quantization_aten_with_scaling:128
  - test/inductor/test_fp8.py (module)
- AOTInductor ABI and export
  - test/inductor/test_aot_inductor.py::test_simple:172
  - test/inductor/test_aot_inductor_package.py::test_add:212
- Performance/metrics
  - test/inductor/test_perf.py::test_pointwise:102
  - test/inductor/test_metrics.py (module)

---

## Suggested Alternate Grouping
When you’re triaging or adding coverage, grouping by “where fixes land” can be handier than the strict pipeline view:

- Capture & Guards
  - control_flow, dynamic_shapes, stance, counters
  - Files: test_control_flow.py, test_torchinductor_dynamic_shapes.py, test_torchinductor_codegen_dynamic_shapes.py
- Graph Normalization
  - functionalization, decompositions, canonicalizations
  - Files: test_auto_functionalize.py, test_decompose_mem_bound_mm.py, test_mmdecomp.py, test_pad_mm.py, test_binary_folding.py, test_inplacing_pass.py
- Post-Grad Shaping
  - custom passes/partitioning/pattern rewriting
  - Files: test_custom_post_grad_passes.py, test_custom_partitioner_fn.py, test_split_cat_fx_passes.py, test_split_cat_fx_aten_passes.py, test_pattern_matcher.py, test_mkldnn_pattern_matcher.py
- IR & Scheduler
  - loop ordering, segmented trees, kernel-level legality
  - Files: test_inductor_scheduler.py, test_loop_ordering.py, test_kernel_optimization.py, test_segmented_tree.py
- Codegen & Backends
  - Triton/CUTLASS/backends, autotune, templates
  - Files: test_triton_kernels.py, test_triton_heuristics.py, test_cutlass_backend.py, test_max_autotune.py, test_codegen_triton.py, test_combo_kernels.py, test_best_config.py
- Runtime & Launch
  - cudagraph trees, static launcher, memory planning
  - Files: test_cudagraph_trees.py, test_static_cuda_launcher.py, test_memory_planning.py, test_device_assert.py
- Caching & Artifacts
  - FXGraph cache, CUDA code cache, remote caches, subprocess
  - Files: test_codecache.py, test_remote_cache.py, test_cudacodecache.py, test_compile_subprocess.py
- Cross-Cutting
  - opinfo/parity, quant/FP8, AOT, metrics/profiling
  - Files: test_torchinductor_opinfo.py, test_op_completeness.py, test_quantization.py, test_fp8.py, test_aot_inductor*.py, test_perf.py, test_metrics.py, test_profiler.py

This grouping tends to mirror how fixes are deployed (Dynamo vs FX passes vs IR vs codegen vs runtime).

---

## Quick Scripts (Isolation Recipes)

These one-liners run focused tests and enable useful logs.

- Capture & guards
  - `pytest -q test/inductor/test_control_flow.py::test_cond_simple_control_flow`
  - `TORCH_LOGS=recompiles pytest -q test/inductor/test_torchinductor_dynamic_shapes.py::test_constant_fold_uniform_value_dynamic`

- Graph normalization
  - `pytest -q test/inductor/test_auto_functionalize.py::test_auto_functionalize_can_with_default`
  - `pytest -q test/inductor/test_decompose_mem_bound_mm.py::test_decompose_bmm`

- Post-grad shaping
  - `TORCH_LOGS=post_grad_graphs pytest -q test/inductor/test_custom_post_grad_passes.py::test_custom_joint_pass_post`
  - `pytest -q test/inductor/test_split_cat_fx_passes.py::test_split_normalization`

- IR & fusion
  - `pytest -q test/inductor/test_loop_ordering.py::TestTiling`
  - `pytest -q test/inductor/test_segmented_tree.py::test_basic_construction`

- Codegen & autotune
  - `pytest -q test/inductor/test_triton_kernels.py::test_triton_kernel_with_kernel_param`
  - `pytest -q test/inductor/test_max_autotune.py::TestMaxAutotune::test_max_autotune_decompose_k_dynamic_input`

- Runtime
  - `pytest -q test/inductor/test_cudagraph_trees.py::test_run_simple`
  - `pytest -q test/inductor/test_static_cuda_launcher.py::test_basic`

- Caching
  - `pytest -q test/inductor/test_codecache.py::TestFxGraphCache::test_cache_guard`
  - `pytest -q test/inductor/test_cudacodecache.py::test_cuda_load`

For convenience, use the helper script: `codex/scripts/inductor_component_tests.sh`.

