# FlashInfer with per-head sparse masks

WaveAlign's FlashInfer experiments require `PerHeadBlockSparseAttentionWrapper` from the [modified FlashInfer repository](https://gitee.com/daidai258/flashinfer). Installing the ordinary `flashinfer-python` package does not provide this wrapper.

## Install the pinned fork

Start with a CUDA/PyTorch environment, then run:

```bash
git clone https://gitee.com/daidai258/flashinfer.git flashinfer-wavealign
cd flashinfer-wavealign
git checkout 9215ce86df41f9b9005eb0c34ff23bdc281327a7
git submodule update --init --recursive
FLASHINFER_LOCAL_VERSION=wavealign pip install .
python -c "from flashinfer import PerHeadBlockSparseAttentionWrapper; print('Per-head sparse backend available')"
```

The package is based on FlashInfer 0.6.11. Keep this installation when setting up WaveAlign; an upgrade to the ordinary PyPI package can replace it.

## Rebuild from upstream plus patch

The included [patch](per-head-sparse.patch) reproduces both fork commits on the exact upstream base. From the WaveAlign repository root:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git ../flashinfer-wavealign
git -C ../flashinfer-wavealign checkout 9f4d16174e81dc818ca65f9eb930ec5fd8a9b395
git -C ../flashinfer-wavealign apply "$PWD/third_party/flashinfer/per-head-sparse.patch"
git -C ../flashinfer-wavealign submodule update --init --recursive
FLASHINFER_LOCAL_VERSION=wavealign pip install ../flashinfer-wavealign
```

The patch adds independent CSR masks per KV head, sequential/concurrent execution, and CUDA device-linking fixes. Q/mask row reordering and output restoration remain the caller's responsibility. Causal attention and internal reordering are explicitly unsupported by this wrapper.

## Hardware configuration

`DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN` uses the WaveAlign integration. Within it, `WAVEALIGN_SPARSE_BACKEND=auto` chooses FA4 on supported Hopper/Blackwell devices when its CuTe module is installed, and the per-head FlashInfer wrapper otherwise. Force either backend with `flashinfer` or `fa4`; requesting FA4 on Ampere raises an explicit error.

Ampere defaults to 64-token sparse blocks; Hopper/Blackwell default to 128. `VLLM_BLK` overrides this. The initial wave-size heuristic is the device SM count, doubled for 64-token blocks; set `WAVEALIGN_WAVE_SIZE` after profiling the actual kernel's occupancy. These defaults are not an occupancy measurement.

`WAVEALIGN_FLASHINFER_SEQUENTIAL=1` executes heads sequentially (default); `0` enables the fork's concurrent path. CUDA reordering extensions target the visible GPUs or an explicit `TORCH_CUDA_ARCH_LIST`, and launch on the input device's current stream.

Batch sizes above one, causal attention, token masks, and unequal Q/K/V shapes bypass the research sparse mask builders and retain the existing dense attention path. The cumulative-RIB variant remains restricted to its SM90 layout. Other platforms retain their original backends; this patch adds no ROCm/XPU/NPU sparse kernels.

GPU correctness has been checked on A100. A40, H100, Blackwell, and multi-GPU numerical/performance validation still require those devices; backend-selection tests alone do not establish hardware validation.

Provenance: upstream `9f4d16174e81dc818ca65f9eb930ec5fd8a9b395`; fork changes `eb82020ae29a2b9c2b4a36077bec7fb40b959dd7` and `9215ce86df41f9b9005eb0c34ff23bdc281327a7`. FlashInfer's Apache-2.0 license and copyright notices apply; the patch retains the original file notices.
