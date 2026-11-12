# GPU validation (2026-11-12)

The physical test device was one NVIDIA A100-SXM4-80GB (SM80, 108 SMs), with CUDA 13.0, PyTorch 2.12.0+cu130, and the [modified FlashInfer](../third_party/flashinfer/README.md) at `9215ce86df41f9b9005eb0c34ff23bdc281327a7`.

## Correctness and portability

- 797 local checks passed, plus the fork's 3 existing tests. Coverage includes row-permutation bijections, determinism, input preservation, FP32/FP16/BF16 attention references, independent batch/head masks, partial K/V blocks, custom softmax scale, CUDA streams, and the centered rank-2 algorithm.
- Source-level integration checks execute the actual forward method and modified FlashInfer kernels. The unavailable vLLM dense dispatcher is adapted to PyTorch SDPA for these checks; this is not a full model/server test.
- Architecture-selection and FA4 API-contract checks use simulated devices. The CUDA greedy extension also built with targets SM80, SM86, SM90, and SM100, and ran on A100.
- Numerical and performance tests on A40, H100, Blackwell, and multiple GPUs remain outstanding. Cross-compilation and routing checks do not establish physical hardware validation. ROCm/XPU/NPU sparse kernels were not added.

The fixes remove the SM90-only extension build, use the input device/current CUDA stream, support the pinned per-head FlashInfer wrapper, avoid zero-padded keys in softmax, preserve dense behavior for unsupported sparse inputs, and restore independent Q/K lengths in dense cross-attention. The main two-dimensional policy now uses centered iteration with QR and preserves wave order during intra-wave NNZ sorting.

## Synthetic-mask performance

Configuration: two heads, head dimension 128, BF16, 64-token blocks, wave size 216, sequential heads. Measurements use warmups and the median of 5 or 9 repetitions. All 84 before/after benchmark cases matched their unreordered sparse outputs exactly.

For the corrected main policy at 436,096 tokens:

| Mask pattern | Prepared kernel speedup | Full attention latency, baseline → reordered | Full-path speedup |
|---|---:|---:|---:|
| Random | 1.111× | 622.605 → 591.383 ms | 1.053× |
| Clustered | 1.249× | 208.952 → 198.658 ms | 1.052× |
| Already ordered clusters | 1.017× | 170.807 → 198.872 ms | 0.859× |

The shorter 8K/32K/131K cases were slower after including overhead. Full-path timing includes reordering, gathers, CSR conversion, planning, backend execution/layout handling, and output restoration. It excludes mask construction and the rest of a video model. Reproduce these measurements with the [standalone benchmark](../benchmarks/wavealign/README.md).

## Limits relative to the paper

The centered iteration currently uses dense batched matrix multiplication; its cost is not the paper's sparse `O(T * nnz)` implementation. The trained adaptive-skip policy is not included in the supplied source, so the measurements force reordering.

No video model weights or real captured masks were available for this run. Wan/LTX end-to-end latency, video-quality metrics, and the paper's H100/A40 figures have not been reproduced. Nsight Compute could not acquire a driver profiling resource, so no measured L2-hit-rate or HBM-traffic claims are made.
