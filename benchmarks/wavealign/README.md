# Sparse attention benchmark

Install PyTorch with CUDA, Triton, and the [pinned FlashInfer fork](../../third_party/flashinfer/README.md), then run from the repository root:

```bash
python benchmarks/wavealign/benchmark_sparse.py \
  --lengths 8192 32768 131072 436096 \
  --methods no_reorder wave_2d_reorder
```

This needs no model weights or full vLLM installation. It measures synthetic masks with independent patterns per head and checks reordered outputs against the unreordered sparse baseline.

Results are written to `benchmarks/wavealign/results/` (ignored by Git). Set `WAVEALIGN_BENCHMARK_OUTPUT` to change the output directory. `--heads`, `--block-size`, and `--wave-size` allow hardware-specific configurations; the default wave size is a heuristic, not measured occupancy.

The report separates prepared kernel latency, reordering, planning, and the complete attention path. The latter includes CSR conversion, Q/mask gathers, backend layout copies, and output restoration, but excludes mask construction and video-model execution. Synthetic-mask speedups do not reproduce the paper's model-level results.
