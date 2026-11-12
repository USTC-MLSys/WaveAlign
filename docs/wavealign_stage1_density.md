# Stage 1 and density investigation

Same-input GPU ablations on the A100 do not identify an inaccurate SVD solver as the main reason for the small speedup. They show limited incremental benefit from Stage 1 on several heads, expensive ordering, and a separate, confirmed XAttention padding defect that affected some previously measured masks.

## Confirmed mask-padding defect

The pinned XAttention noncausal path included zero-padded keys in its estimator softmax and cumulative block selection. Discarding those padding columns afterward could leave real keys far below the requested 90% estimated attention mass. This was present in both the Triton and PyTorch paths.

An independent constant-negative-logit example at 4,160 tokens retained only 1.54% of real-key estimated mass before the fix. After excluding padding, it retained 90.77%, as expected for discrete block selection. Aligned lengths and the checked causal cases were unchanged.

On the **same captured Wan2.2 Q/K/V** at step 0, layer 24, 147,600 tokens, head 20 changed from density **0.393% to 10.607%**. Its real-key estimated coverage returned to at least 90%. The average over all 40 heads changed only from **69.732% to 70.137%**, illustrating why the mean alone hid the problem.

The fix is supplied in [noncausal-padding.patch](../third_party/xattention/noncausal-padding.patch). Both the native integration and the public benchmark now pass original Q/K lengths to the estimator; XAttention handles its own storage padding. The validation patch does not change Stage 1 or tune the threshold to produce a desired density.

## Stage 1 ablation on fixed masks

Four representative heads (0, 2, 20, 33) were selected from the original head-density quantiles and the largest row-NNZ CV. All candidates then used identical Q/K/V and corrected masks. The figures below sum each head's median **prepared FA2 execution** time; they are not 40-head totals or video latency. Planning and Q/mask preparation are excluded from this kernel comparison.

| Policy | Prepared execution, four heads |
|---|---:|
| Original order | 148.718 ms |
| Global NNZ sort | 142.614 ms |
| Stage 2 only | 143.788 ms |
| Current Stage 1 + Stage 2 | 143.157 ms |
| 30-iteration Stage 1 + Stage 2 | 143.309 ms |
| Exact centered SVD + Stage 2 | 143.510 ms |
| Raw u2/u3 + the same Stage 2 | 143.854 ms |
| Cumulative-SVD/RIB + the same Stage 2 | 142.796 ms |

Current Stage 1 saved about **0.632 ms** relative to Stage 2 alone. Computing the permutations increased from **0.552 ms to 10.268 ms** across these heads. The additional ordering cost therefore outweighed the observed kernel saving, before accounting for gathers and output restoration.

The ten-iteration centered basis had **97.4%–99.97% subspace overlap** with the exact top-two subspace (mean squared principal-angle cosines). More accurate bases did not consistently improve execution time. This is not proof that every Stage 1 method is optimal: individual heads differ, and a different wave boundary or embedding can improve one head while worsening another.

The centered rank-two projection explained about 17.7%–27.6% of matrix energy in these sampled heads. For the dense heads, Stage 1 reduced pairwise mask distance but each wave still touched almost all 2,307 K/V blocks. Head 20 benefited more: the mean wave union fell from about 2,285 to 1,788 blocks, and its kernel improved. These are mask-geometry statistics, not measured cache hit rates.

## Density changes through denoising

With corrected masks, a real 40-step schedule was executed through step index 15 at 480×832 / 81 frames (32,760 tokens). Conditional-path samples from 40 heads and three layers at four steps produced 480 head records; all met the 90% estimated-mass condition within numerical tolerance.

| Step index | Expert | Layer 24 mean density |
|---:|---|---:|
| 0 | High noise | 65.85% |
| 3 | High noise | 60.20% |
| 7 | High noise | 59.21% |
| 15 | Low noise | 56.15% |

Density is not fixed by `threshold=0.9`: the threshold controls cumulative estimated attention mass. A diffuse score distribution can require a dense mask. These Wan2.2 measurements are not the same workload as the paper's Wan2.1-14B/H100 step-16/layer-24 characterization, which reports XAttention density around 32%.

Additional same-input ablations on the captured late-step heads also showed mixed Stage 1 benefits. Exact SVD improved some kernels but its ordering cost was much larger; raw u2/u3 and cumulative-RIB did not remove the general cost/benefit problem. This supports evaluating reordering profitability per workload/head, rather than assuming that changing the SVD solver alone will recover the paper's speedup.

## Validation and scope

- 29 estimator GPU checks cover negative logits, partial lengths, unequal Q/K lengths, FP16/BF16, Triton/PyTorch implementations, threshold monotonicity, and an independent masked-softmax reference.
- Combined estimator and integration regression: 54 checks passed. Five comparisons against saved pre-fix code confirmed unchanged causal/aligned behavior.
- A paired real-model regression exercised both Wan2.2 experts with the corrected estimator and preserved the baseline/reordered latents exactly. The native Wan2.1 vLLM-Omni entrypoint also generated a video successfully with the patched estimator.
- Historical long-sequence measurements retain their original masks and are labeled accordingly. Corrected full-video quality and end-to-end speedups at the earlier longest lengths have not been re-established by this investigation.

The production Stage 1 policy was kept unchanged while fixing mask selection. The current code still uses dense BMM/QR and has no frozen, trained adaptive-skip policy; neither is equivalent to the paper's complete optimized runtime.
