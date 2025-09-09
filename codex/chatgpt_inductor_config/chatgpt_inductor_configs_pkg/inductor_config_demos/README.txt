Inductor config demos

Demos included:
- demo_fx_graph_cache.py: Show FX graph cache hits/misses
- demo_worker_threads.py: Adjust worker/thread-related settings (if applicable)
- demo_fusion_sandbox.py: Toggle fusion heuristics and compare kernel counts + fusion logs

Examples:
  # Compare fusion strength on pointwise chain
  python demo_fusion_sandbox.py --model pointwise \
    --baseline max_fusion_size=64 \
    --variant max_fusion_size=1,score_fusion_memory_threshold=1000,realize_acc_reads_size_threshold=1024,pick_loop_orders=False

  # Probe epilogue fusion on GEMM+ReLU
  python demo_fusion_sandbox.py --model mmrelu

  # Run multiple variants and save CSV
  python demo_fusion_sandbox.py --model pointwise \
    --variants 'max_fusion_size=8|max_fusion_size=1,pick_loop_orders=False' \
    --csv fusion_results.csv
