# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-aware execution of full-block, noncausal sparse attention.

The caller reorders Q and mask rows and restores the output. CUDA devices
without FA4 support use the pinned FlashInfer per-head wrapper. Optional
backends are imported only when needed, so importing other platform backends
does not require a Hopper-only package.
"""

import importlib
import math
import os

import torch

_FLASHINFER_HELP = (
    "Install the pinned WaveAlign FlashInfer fork; the standard PyPI package "
    "does not provide PerHeadBlockSparseAttentionWrapper. See "
    "https://github.com/USTC-MLSys/WaveAlign/tree/main/third_party/flashinfer"
)


def choose_backend(capability: tuple[int, int], requested: str, fa4_available: bool) -> str:
    """Select without treating installed FA4 as support for Ampere hardware."""
    if requested not in {"auto", "fa4", "flashinfer"}:
        raise ValueError("WAVEALIGN_SPARSE_BACKEND must be auto, fa4, or flashinfer")
    supports_fa4 = capability[0] in {9, 10}
    if requested == "fa4":
        if not supports_fa4:
            raise ValueError(f"FA4 requires supported Hopper/Blackwell hardware; got SM{capability[0]}{capability[1]}")
        if not fa4_available:
            raise ImportError("FA4 was requested but flash_attn.cute is unavailable")
        return "fa4"
    if requested == "auto" and supports_fa4 and fa4_available:
        return "fa4"
    return "flashinfer"


def _csr_for_head(counts, indices, kv_length: int, kv_block: int):
    """Preserve a partial final K/V block without adding zero keys to softmax.

    FlashInfer needs N divisible by its page size. Subdivide mask columns using
    gcd(N, C) when necessary, retaining only actual key positions.
    """
    page_size = math.gcd(kv_length, kv_block)
    subdivisions = kv_block // page_size
    slots = torch.arange(indices.shape[-1], device=indices.device)
    valid = slots[None, :] < counts[:, None]
    if subdivisions > 1:
        offsets = torch.arange(subdivisions, device=indices.device)
        pages = (indices[:, :, None] * subdivisions + offsets).flatten(1)
        valid = valid[:, :, None].expand(-1, -1, subdivisions).flatten(1)
    else:
        pages = indices
    valid = valid & (pages < kv_length // page_size)
    lengths = valid.sum(1, dtype=torch.int32)
    indptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    return indptr, pages[valid].to(torch.int32).contiguous(), page_size


def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    full_block_cnt: torch.Tensor,
    full_block_idx: torch.Tensor,
    block_size: tuple[int, int],
    softmax_scale: float | None = None,
    backend: str | None = None,
    sequential_heads: bool | None = None,
) -> torch.Tensor:
    """Run independent masks per batch/head on [batch, sequence, head, dim]."""
    if query.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("WaveAlign sparse kernels currently require an NVIDIA CUDA GPU")
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Q/K/V must have [batch, sequence, head, dim] shapes, with matching K/V")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("This sparse path requires equal Q/K/V batch sizes, head counts and head dimensions")
    if any(t.device != query.device or t.dtype != query.dtype for t in (key, value)):
        raise ValueError("Q/K/V must have matching devices and dtypes")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("Sparse CUDA attention requires float16 or bfloat16")
    batch, length, heads, dim = query.shape
    rows, columns = block_size
    if min(batch, length, heads, dim, key.shape[1], rows, columns) <= 0:
        raise ValueError("Tensor dimensions and block sizes must be positive")
    expected = (batch, heads, (length + rows - 1) // rows)
    if full_block_cnt.shape != expected or full_block_idx.shape[:3] != expected or full_block_idx.ndim != 4:
        raise ValueError("Sparse metadata must have [batch, head, Q block] counts and matching column indices")
    if any(t.device != query.device or t.dtype != torch.int32 for t in (full_block_cnt, full_block_idx)):
        raise ValueError("Sparse metadata must be int32 on the Q/K/V device")

    requested = backend or os.environ.get("WAVEALIGN_SPARSE_BACKEND", "auto")
    with torch.cuda.device(query.device):
        capability = torch.cuda.get_device_capability(query.device)
        fa4 = None
        if requested != "flashinfer" and capability[0] in {9, 10}:
            try:
                fa4 = importlib.import_module("flash_attn.cute.interface")
            except (ImportError, ModuleNotFoundError):
                pass
        selected = choose_backend(capability, requested, fa4 is not None)
        if selected == "fa4":
            sparse = dict(
                full_block_cnt=full_block_cnt,
                full_block_idx=full_block_idx,
                mask_block_cnt=torch.zeros_like(full_block_cnt),
                mask_block_idx=torch.zeros_like(full_block_idx),
                block_size=block_size,
            )
            # The high-level FA4 function no longer accepts expanded sparse
            # keyword arguments. The structured low-level interface supports
            # both 64- and 128-token tiles without depending on that legacy API.
            tensors = importlib.import_module("flash_attn.cute.block_sparsity").BlockSparseTensorsTorch
            out = fa4._flash_attn_fwd(
                query,
                key,
                value,
                softmax_scale=softmax_scale,
                tile_mn=block_size,
                block_sparse_tensors=tensors(**sparse),
            )
            return out[0] if isinstance(out, tuple) else out

        try:
            wrapper_class = importlib.import_module("flashinfer").PerHeadBlockSparseAttentionWrapper
        except (ImportError, AttributeError) as exc:
            raise ImportError(_FLASHINFER_HELP) from exc
        if sequential_heads is None:
            sequential_heads = os.environ.get("WAVEALIGN_FLASHINFER_SEQUENTIAL", "1") == "1"
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=query.device)
        wrapper = wrapper_class(workspace, backend="fa2" if capability[0] < 9 else "auto")
        outputs = []
        for item in range(batch):
            csr = [
                _csr_for_head(full_block_cnt[item, head], full_block_idx[item, head], key.shape[1], columns)
                for head in range(heads)
            ]
            wrapper.plan(
                [entry[0] for entry in csr],
                [entry[1] for entry in csr],
                length,
                key.shape[1],
                rows,
                csr[0][2],
                heads,
                heads,
                dim,
                sequential_heads=sequential_heads,
                sm_scale=softmax_scale,
                q_data_type=query.dtype,
                kv_data_type=key.dtype,
                o_data_type=query.dtype,
            )
            outputs.append(wrapper.run(query[item].contiguous(), key[item].contiguous(), value[item].contiguous()))
        return torch.stack(outputs)
