# XAttention mask estimator

The real-model benchmark uses the official XAttention estimator at commit `e37988770b9d1bebd489eba011d615f35587ba08`. Its separate `block_sparse_attn` execution backend is unnecessary when WaveAlign executes the mask with FlashInfer.

From the WaveAlign root, install the estimator with both patches:

```bash
git clone https://github.com/mit-han-lab/x-attention.git ../x-attention
git -C ../x-attention checkout e37988770b9d1bebd489eba011d615f35587ba08
git -C ../x-attention apply "$PWD/third_party/xattention/lazy-execution-import.patch"
git -C ../x-attention apply "$PWD/third_party/xattention/noncausal-padding.patch"
pip install --no-deps -e ../x-attention
```

The lazy-import patch changes when the optional execution library is imported. Calling XAttention's own execution function still requires that library.

The noncausal-padding patch excludes padding keys from the estimator's softmax and block selection in both Triton and PyTorch paths. The pinned upstream version otherwise treats chunk-padding zeros as valid keys; for negative Q/K logits this can select padding blocks and leave the real keys far below the requested cumulative attention mass. Call the estimator with the original Q/K lengths; it manages its own storage padding. This fix changes affected masks and requires refreshing results obtained with the old estimator.

GPU checks cover uneven lengths, unequal query/key lengths, FP16/BF16, both implementations, an independent masked-softmax reference, and threshold monotonicity.
