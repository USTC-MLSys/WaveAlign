"""Standalone benchmark helpers; imports the source operators without vLLM initialization."""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("WAVEALIGN_BENCHMARK_OUTPUT", Path(__file__).resolve().parent / "results")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
BACKENDS = ROOT / "vllm_omni/diffusion/attention/backends"


def load_source(name):
    spec = importlib.util.spec_from_file_location("validation_" + name, BACKENDS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


greedy = load_source("greedy_reorder_cuda")
rib = load_source("cumulative_svd_rib")


def policies(wave_size=132):
    return {
        "no_reorder": lambda m: torch.arange(m.shape[1], device=m.device).expand(m.shape[0], -1).contiguous(),
        "nnz_reorder": lambda m: m.sum(-1).argsort(dim=1, descending=True),
        "wave_1d_reorder": lambda m: greedy.wave_1d_reorder(m.float(), wave_size),
        "wave_2d_reorder": lambda m: greedy.wave_2d_reorder(m.float(), wave_size),
        "wave_reorder": lambda m: greedy.wave_overlap_reorder(m.float(), wave_size),
        "svd_u2u3_reorder": lambda m: greedy.svd_u2u3_reorder(m.float(), wave_size),
        "svd_deflation_reorder": lambda m: greedy.svd_deflation_reorder(m.float(), wave_size),
        "hybrid_reorder": lambda m: greedy.hybrid_block_reorder(m.float(), wave_size),
        "nnz_3step": lambda m: greedy.nnz_3step_reorder(m.float(), wave_size),
        "svd_cumulative_rib_reorder": lambda m: rib.reorder(m.bool().contiguous()),
        "nnz_head_aligned_reorder": lambda m: rib.reorder(m.bool().contiguous(), stage2_only=True),
    }


def make_mask(heads, rows, cols, pattern="random", density=0.3, seed=42, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (heads, rows, cols)
    if pattern == "zero":
        return torch.zeros(shape, device=device, dtype=torch.bool)
    if pattern == "dense":
        return torch.ones(shape, device=device, dtype=torch.bool)
    if pattern == "single":
        mask = torch.zeros(shape, device=device, dtype=torch.bool)
        mask[..., 0] = True
        return mask
    if pattern == "topk":
        scores = torch.rand(shape, generator=gen, device=device)
        idx = scores.topk(max(1, round(cols * density)), dim=-1).indices
        return torch.zeros(shape, device=device, dtype=torch.bool).scatter_(-1, idx, True)
    if pattern in ("clustered", "ordered_clusters"):
        clusters = min(8, max(2, cols // 16))
        row_groups = torch.arange(rows, device=device) % clusters
        if pattern == "ordered_clusters":
            row_groups = torch.arange(rows, device=device) * clusters // rows
        centers = torch.arange(cols, device=device) * clusters // cols
        same = row_groups[:, None] == centers[None, :]
        probs = torch.where(same, min(0.95, density * clusters), density * 0.08)
        mask = torch.rand(shape, generator=gen, device=device) < probs
    elif pattern == "variable":
        probabilities = torch.linspace(0.02, min(1.0, 2 * density), rows, device=device)
        mask = torch.rand(shape, generator=gen, device=device) < probabilities[None, :, None]
    else:
        mask = torch.rand(shape, generator=gen, device=device) < density
    # Real softmax rows must select at least one key.
    mask[..., 0] = True
    return mask.contiguous()


def verify_permutation(perm, mask):
    expected = torch.arange(mask.shape[1], device=mask.device).expand(mask.shape[0], -1)
    assert perm.shape == mask.shape[:2]
    assert perm.dtype == torch.int64 and perm.device == mask.device
    assert torch.equal(perm.sort(1).values, expected), "not a per-head bijection"


def cuda_timing(fn, warmup=3, repeat=11):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    device_ms, wall_ms = [], []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - wall) * 1000)
        device_ms.append(start.elapsed_time(end))
    return {
        "gpu_median_ms": statistics.median(device_ms),
        "wall_median_ms": statistics.median(wall_ms),
        "gpu_min_ms": min(device_ms),
        "gpu_max_ms": max(device_ms),
        "samples_ms": device_ms,
    }


def dump_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, default=str) + "\n")


def independent_attention(q, k, v, mask, block):
    """Double-precision dense reference for small cases, [B,H,L,D] tensors."""
    expanded = mask.repeat_interleave(block, 1).repeat_interleave(block, 2)
    score = q.double() @ k.double().transpose(-1, -2) / q.shape[-1] ** 0.5
    score.masked_fill_(~expanded[None], -torch.inf)
    probability = torch.nan_to_num(score.softmax(-1))
    return probability @ v.double()


def reorder_q(q, perm, block):
    # q is [B,H,L,D]. Each head has its own block permutation.
    b, h, length, dim = q.shape
    idx = perm[None, :, :, None, None].expand(b, h, length // block, block, dim)
    return q.reshape(b, h, length // block, block, dim).gather(2, idx).reshape_as(q).contiguous()


def restore_output(out, perm, block):
    b, h, length, dim = out.shape
    idx = perm[None, :, :, None, None].expand(b, h, length // block, block, dim)
    restored = torch.empty_like(out).reshape(b, h, length // block, block, dim)
    restored.scatter_(2, idx, out.reshape_as(restored))
    return restored.reshape_as(out)
