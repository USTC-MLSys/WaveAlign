# XAttention mask estimator

The real-model benchmark uses the official XAttention estimator at commit `e37988770b9d1bebd489eba011d615f35587ba08`. Its separate `block_sparse_attn` execution backend is unnecessary when WaveAlign executes the mask with FlashInfer.

From the WaveAlign root, install the estimator with the small lazy-import patch:

```bash
git clone https://github.com/mit-han-lab/x-attention.git ../x-attention
git -C ../x-attention checkout e37988770b9d1bebd489eba011d615f35587ba08
git -C ../x-attention apply "$PWD/third_party/xattention/lazy-execution-import.patch"
pip install --no-deps -e ../x-attention
```

The patch changes only when the optional execution library is imported. Mask estimation and its Triton kernels are unchanged. Calling XAttention's own execution function still requires that library.
