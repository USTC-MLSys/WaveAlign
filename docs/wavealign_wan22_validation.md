# Wan2.2 A14B: sequence-length and GPU-capacity validation

**Historical estimator configuration:** these measurements preceded the [noncausal XAttention padding fix](../third_party/xattention/README.md). Output-equivalence checks remain valid for those shared masks, but affected densities and model outputs must be refreshed before treating them as results for the corrected estimator. See the [Stage 1 and density investigation](wavealign_stage1_density.md).

Actual Wan2.2 weights were tested on one A100-SXM4-80GB. Increasing sequence length changes the balance between kernel savings and reordering overhead, but neither model size nor sequence length guarantees a net speedup for every mask.

## Sparse method and kernel

The mask estimator is XAttention with threshold 0.9, 64×64 mask blocks, stride 16, and retained diagonal blocks. Execution uses the [pinned modified FlashInfer](../third_party/flashinfer/README.md), `PerHeadBlockSparseAttentionWrapper`, **FA2** on A100, and sequential heads. The model has 40 heads with head dimension 128; the ordinary sweep groups eight heads to bound workspace. WaveAlign uses centered rank-2 iteration with QR and intra-wave NNZ ordering, with wave size 216.

The 720p lengths below are not divisible by 64. The adapter preserves the real K/V length by subdividing KV pages to 16 tokens. A 64×64 mask therefore does not imply an identical kernel configuration to the paper's original 64×64 setup.

## Real attention at increasing lengths

The high-noise expert actually runs through the preceding blocks before Q/K/V and masks are captured at denoising step 0, layer index 24. Each policy uses the same captured tensors. Timings are medians of five repetitions after warmup, including reordering, metadata, CSR/planning, execution and output restoration; mask estimation and the rest of the model are excluded.

| Tokens | Frames at 720p | Mask density | Original | NNZ sort | WaveAlign | Original / WaveAlign |
|---:|---:|---:|---:|---:|---:|---:|
| 75,600 | 81 | 0.665 | 0.524 s | 0.507 s | 0.584 s | 0.898× |
| 147,600 | 161 | 0.697 | 2.059 s | 1.960 s | 2.079 s | 0.991× |
| 291,600 | 321 | 0.701 | 8.383 s | 7.786 s | 7.853 s | 1.067× |
| 363,600 | 401 | 0.704 | 13.406 s | 12.512 s | 12.397 s | 1.081× |

All outputs matched exactly. NNZ sorting was slightly faster than WaveAlign at the first three points. Mask structure also changes with video length, so this curve supports length sensitivity without isolating length as the sole cause. No L2 hit-rate or HBM-traffic counters were available.

## Largest checked sequence

Using CPU offload, one head per temporary workspace group, and expandable CUDA allocator segments, the actual first complete Transformer block repeatedly ran at **486,000 tokens** (537 frames at 720p), with a peak allocated GPU memory of **78.42 GiB**. The next valid frame-count increment, **489,600 tokens** (541 frames), failed with CUDA OOM. Larger probes at 496,800, 525,600 and 579,600 also failed.

This capacity probe executes the first block's 40-head self-attention, cross-attention and feed-forward network, then captures layer 1. It does not establish that a full 537-frame video, every denoising step, or the low-noise expert completes at that length. The bound depends on the allocator, grouping, runtime version and surrounding allocations.

The captured first head at 486K is particularly sparse (density 0.00287), and WaveAlign was slower for that head despite the long sequence. Its outputs remained exact, and an independent sampled-query FP32 reference agreed within BF16 tolerance. Different layers/heads must not be mixed into the layer-24 curve above.

Replaying **all 40 heads** of that 486K-token layer produced the following medians after warmup (three measured repetitions, one head per workspace group):

| Original | NNZ sort | WaveAlign | Original / WaveAlign |
|---:|---:|---:|---:|
| 13.555 s | 13.418 s | 12.679 s | 1.069× |

All outputs matched exactly. Mean mask density was 0.3913 and the effective KV page was 16. This is layer 1 with head grouping 1, so it remains separate from the layer-24/group-8 curve. It confirms a net attention-path benefit at the largest checked model length, not full-video speedup at that length.

## Complete two-expert generation

A 720×1280, 161-frame, four-step run exercised both experts: 160 attention calls per expert, 320 total, with 1,600 head-group executions. The original sparse path took **949.589 s** and WaveAlign took **950.140 s**. Final latents and decoded uint8 pixels matched exactly. This single-run difference is effectively a tie, not evidence of an end-to-end improvement.

Reordering/metadata grew from 3.338 s to 42.427 s, while sparse-backend time fell from 501.906 s to 469.778 s. Whole-generation timings include CPU/GPU offload and VAE decode, but exclude initial checkpoint loading, precomputed prompt embeddings and MP4 encoding. Four steps are a correctness/performance diagnostic, not a converged-quality evaluation.

## Provenance and reproduction

The official checkpoint revision was `5be7df9619b54f4e2667b2755bc6a756675b5cd7`; each expert has 14,288,491,584 parameters. All 24 original expert shards were SHA256-verified and converted from FP32 to BF16. With only a 30GB filesystem, the approximately 53.2 GiB of BF16 expert weights were temporarily held in RAM-backed files, then retained by the loaded model process. Shared UMT5 values were sampled against the official checkpoint; the VAE file hash matches the official Wan2.2 repository.

The runtime was Diffusers 0.36.0, PyTorch 2.12.0+cu130, the pinned FlashInfer fork `9215ce8`, guidance 4/3, seed 42, and the official scheduler configuration. This is actual model execution through a Diffusers adapter using the repository's sparse operators. It is not a native vLLM-Omni Wan2.2 serving run or a reproduction of the paper's H100/A40 results.

The [benchmark](../benchmarks/wavealign/README.md#wan22-a14b) now instruments both experts and supports CPU offload and head grouping. Its real-model regression executed both experts and preserved the outputs exactly. The main reordering implementation still uses dense BMM/QR rather than the paper's sparse `O(T * nnz)` operator, and the supplied source contains no trained adaptive-skip tree.
