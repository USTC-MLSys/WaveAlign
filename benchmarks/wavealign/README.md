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

## Real Wan video generation

Install the [XAttention estimator](../../third_party/xattention/README.md) and pinned FlashInfer fork, plus `diffusers==0.36.0`, `transformers==4.57.6`, `huggingface_hub==0.36.0`, `tokenizers>=0.22,<0.23`, `accelerate==1.12.0`, `sentencepiece`, `ftfy`, and `imageio[ffmpeg]`. Put the active environment's `bin` directory on `PATH` so the CUDA JIT can find `ninja`.

```bash
python benchmarks/wavealign/benchmark_wan.py \
  --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --height 480 --width 832 --frames 81 --steps 30 \
  --modes dense no_reorder wave_2d_reorder --capture
```

This runs an actual pretrained Diffusers Wan pipeline, replacing self-attention with XAttention masks and this repository's sparse execution/reordering operators. It is a real-model adapter, not the native vLLM-Omni serving entrypoint. Cross-attention retains Diffusers' normal implementation.

Outputs include MP4s, latents, decoded frames, timings, and exact output comparisons. `--capture` saves real Q/K/V and the mask at step 16/layer 24 (or the last step of a shorter run); its extra I/O time is reported separately. `--save-float-frames` retains full float frames; the default stores uint8 frames to reduce disk use. Model loading, prompt encoding, and video-file encoding are excluded from generation latency; denoising and VAE decoding are reported separately. Run the benchmark alone on the GPU for timing comparisons.
