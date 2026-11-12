"""
CPU-based greedy nearest-neighbor row reordering.

Optimized for large block counts:
  - GPU torch.mm for intersection matrix (faster than CPU BLAS for large MB)
  - Numba JIT compiled greedy loop (eliminates Python overhead)
  - Supports Jaccard / Hamming / Intersection metrics

Dependencies: torch, numpy, numba. No custom CUDA / C++ extensions.
"""

import numpy as np
import torch
import numba


# ---------------------------------------------------------------------------
# Numba JIT compiled greedy nearest-neighbor (no Python loop overhead)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _greedy_nn_jit(sim):
    """Greedy nearest-neighbor chain, compiled with Numba for speed."""
    MB = sim.shape[0]
    perm = np.empty(MB, dtype=np.int64)
    visited = np.zeros(MB, dtype=numba.boolean)

    current = 0
    perm[0] = current
    visited[current] = True

    for step in range(1, MB):
        best_idx = -1
        best_val = -1e30
        for j in range(MB):
            if not visited[j] and sim[current, j] > best_val:
                best_val = sim[current, j]
                best_idx = j
        perm[step] = best_idx
        visited[best_idx] = True
        current = best_idx

    return perm


# Warm up numba JIT on import (small matrix, ~200ms one-time cost)
_greedy_nn_jit(np.zeros((2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def reorder_rows_by_similarity(
    mask,
    metric="intersection",
):
    """
    Greedy nearest-neighbor row reordering.

    Parameters
    ----------
    mask : torch.Tensor [MB, NB], int32, on CUDA or CPU
    metric : "jaccard" | "hamming" | "intersection"

    Returns
    -------
    perm : torch.Tensor [MB], int64
    """
    device = mask.device
    MB, NB = mask.shape

    if MB <= 1:
        return torch.arange(MB, device=device)

    # --- Step 1: GPU matmul for intersection (faster than CPU BLAS) --------
    mask_f = mask.float()  # ensure float for matmul
    with torch.amp.autocast(device_type='cuda', enabled=False):
        intersection = (mask_f @ mask_f.T)  # [MB, MB], always float32

    # --- Step 2: Compute similarity metric --------------------------------
    if metric == "intersection":
        sim = intersection
    elif metric == "jaccard":
        row_nnz = mask_f.sum(dim=1)  # [MB]
        union = row_nnz.unsqueeze(1) + row_nnz.unsqueeze(0) - intersection
        sim = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
    elif metric == "hamming":
        row_nnz = mask_f.sum(dim=1)
        hamming_dist = row_nnz.unsqueeze(1) + row_nnz.unsqueeze(0) - 2 * intersection
        sim = -hamming_dist  # negate: higher = more similar
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    sim.fill_diagonal_(-float('inf'))  # no self-loops

    # --- Step 3: Transfer to CPU numpy + Numba JIT greedy loop ------------
    sim_np = sim.cpu().numpy().astype(np.float32)
    perm_np = _greedy_nn_jit(sim_np)

    return torch.tensor(perm_np, dtype=torch.int64, device=device)


def graph_mst_reorder(mask_int, metric="intersection", traversal="dfs"):
    """Drop-in wrapper for benchmark_all.py."""
    return reorder_rows_by_similarity(mask_int, metric=metric)
