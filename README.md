# WaveAlign

**Cache-Aware Query-Row Scheduling for Sparse Attention in Long-Video Generation**

WaveAlign improves K/V cache locality by grouping query rows with similar access patterns and scheduling them within GPU waves. It permutes query blocks and sparse-mask rows before attention, then restores the original output order.

This repository contains research code built on [vLLM-Omni](https://github.com/vllm-project/vllm-omni).

## Installation

Prepare a Linux environment with Python 3.12, compatible CUDA/PyTorch and vLLM 0.20.0. XAttention must be importable as `xattn.src.Xattention`; install SpargeAttn separately if using `SPARSE_MODE=sparge`.

Sparse execution selects FlashAttention 4 on supported Hopper/Blackwell devices when installed, and the [pinned per-head FlashInfer fork](third_party/flashinfer/README.md) otherwise, including A100/A40. See [hardware configuration](third_party/flashinfer/README.md#hardware-configuration) for overrides and validation limits.

```bash
git clone https://github.com/USTC-MLSys/WaveAlign.git
cd WaveAlign
pip install numba
VLLM_OMNI_TARGET_DEVICE=cuda VLLM_OMNI_VERSION_OVERRIDE=0.20.0+wavealign pip install -e .
```

## Usage

Example using the 2D spectral reordering policy and two compatible NVIDIA GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
SPARSE_MODE=xattn VLLM_THRESHOLD=0.9 VLLM_MODE=wave_2d_reorder \
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 40 --tensor-parallel-size 2 \
  --output wavealign.mp4
```

Use `VLLM_MODE=no_reorder` for the sparse baseline or `VLLM_MODE=nnz_reorder` for sorting by nonzero block count. GPU memory requirements depend on the model and video size; this example has not been validated as part of this release.

## Code

- [Reordering algorithms](vllm_omni/diffusion/attention/backends/greedy_reorder_cuda.py)
- [Attention integration](vllm_omni/diffusion/attention/backends/flash_attn.py)
- [Video generation and profiling](examples/offline_inference/text_to_video/)
- [GPU validation and limitations](docs/wavealign_validation.md)

## Acknowledgments and License

Built on vLLM-Omni, vLLM, FlashAttention, XAttention, and SpargeAttn. See the [original vLLM-Omni README](README.upstream.md) for upstream documentation and citation.

Licensed under [Apache-2.0](LICENSE). Upstream copyright and license notices are retained.
