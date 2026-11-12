# Real-model GPU validation (2026-11-12)

**Historical estimator configuration:** these runs preceded the [noncausal XAttention padding fix](../third_party/xattention/README.md). Their paired output comparisons use the same pre-fix masks; they do not verify the corrected mask selection. See the [subsequent investigation](wavealign_stage1_density.md).

Actual pretrained Wan2.1-T2V-1.3B inference was run on one NVIDIA A100-SXM4-80GB, including the native vLLM-Omni entrypoint. **Reordering preserved the outputs exactly, but the current implementation did not improve total generation latency on these configurations.**

## Results

| Execution path | Video / steps | Sparse baseline | WaveAlign | Baseline / WaveAlign |
|---|---|---:|---:|---:|
| Native vLLM-Omni | 480×832, 81 frames, 30 steps | 103.122 s | 122.484 s | 0.842× |
| Diffusers with WaveAlign operators | 480×832, 81 frames, 30 steps | 108.098 s | 128.808 s | 0.839× |
| Diffusers with WaveAlign operators | 720×1280, 121 frames, 16 steps | 327.119 s | 349.352 s | 0.936× |

Each row uses the same prompt, seed, model weights, mask policy and sampling configuration for its two runs. These are single-run observations, not confidence-interval estimates. The native entrypoint includes prompt processing; the Diffusers measurements precompute prompt embeddings. Do not compare different execution paths as if all settings and timing boundaries were identical.

Both native runs executed 1,800 sparse attention calls. Their saved float32 frames, captured before MP4 encoding, were identical (maximum absolute error zero). Both Diffusers pairs also had identical final latents; the 480p float frames and 720p stored uint8 frames matched exactly. The sequence lengths were 32,760 and 111,600 tokens, respectively.

The Diffusers 480p dense baseline took 115.747 s. Dense and sparse generation produced different videos (PSNR 9.277 dB for this particular pair); exact WaveAlign equivalence is relative to the same sparse baseline, not dense attention. These observations do not establish VBench quality or reproduce the paper's full workload.

## Setup and evidence

- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, revision `0fad780a534b6463e45facd96134c9f345acfa5b`. Transformer and UMT5 weights were cast from FP32 to BF16 on disk to fit the available storage; the VAE remained FP32. Stored weights occupy 13.697 GiB.
- Native runtime: vLLM 0.20.0, PyTorch 2.11.0+cu130, TP=1, eager execution. The actual repository scheduler, worker, model loader, and attention backend ran without mocked model or kernel components.
- Adapter runtime: Diffusers 0.36.0, PyTorch 2.12.0+cu130. The adapter executes real pretrained model layers with the repository's reordering/sparse dispatch operators.
- Both paths: pinned [per-head FlashInfer fork](../third_party/flashinfer/README.md), [XAttention estimator](../third_party/xattention/README.md), threshold 0.9, 12 heads, head dimension 128, 64-token blocks, wave size 216, sequential heads, guidance 6, seed 42.
- Real Q/K/V and masks were captured at denoising step 16 (index 15), layer index 24 in both adapter runs. Three replay policies (original, NNZ, WaveAlign) returned identical outputs. Sampled queries checked against an independent FP32 reference had maximum absolute error 0.008243 / 0.008533, within the BF16 tolerance.
- Complete attention replay medians (including preparation and reordering) were 32.408 / 25.579 / 38.380 ms at 480p, and 188.278 / 174.901 / 209.843 ms at 720p, for original / NNZ / WaveAlign.

Model loading and MP4 encoding are excluded from reported generation time. Adapter timings include denoising and VAE decode. Capturing real tensors added 0.355 s / 1.203 s to the adapter baseline; excluding that I/O gives 107.743 s / 325.915 s and leaves the conclusion unchanged. The 16-step long-video run is a correctness/performance check, not a converged-quality claim.

## Remaining performance gap

Reordering/metadata added approximately 21.30 s at 480p and 31.42 s at 720p, while backend execution saved only approximately 0.66 s and 8.02 s. The current centered iteration uses dense batched products and QR, rather than the paper's sparse `O(T * nnz)` operator, and the supplied source contains no trained adaptive-skip tree. These forced-reorder results must not be presented as reproducing the paper's end-to-end speedup.

No A40/H100/Blackwell or multi-GPU numerical tests were performed. Nsight Compute counters remained unavailable, so no measured L2-hit-rate or HBM-traffic conclusion is made. The old native MFU log assumes an H100/H200 peak and is not used to evaluate A100.

## Reproduce

The [real-model benchmark instructions](../benchmarks/wavealign/README.md#real-wan-video-generation) cover the adapter and checkpoint setup. For the native path, install vLLM before the pinned FlashInfer fork so dependency installation does not replace it, then run the example twice with `VLLM_MODE=no_reorder` and `VLLM_MODE=wave_2d_reorder`:

```bash
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
WAVEALIGN_SPARSE_BACKEND=flashinfer SPARSE_MODE=xattn \
VLLM_THRESHOLD=0.9 VLLM_MODE=wave_2d_reorder \
python examples/offline_inference/text_to_video/text_to_video.py \
  --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 30 \
  --tensor-parallel-size 1 --guidance-scale 6 --seed 42 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --enforce-eager --vae-use-tiling --output wavealign.mp4
```
