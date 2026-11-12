# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full cumulative rank-2 SVD and capacity-constrained row grouping.

The nominal launch buckets follow an x-fastest (Q blocks, heads, batch) grid.
They are not hardware synchronization barriers. Callers must use batch size 1,
equal Q/KV heads and a non-persistent SM90 B128 FA4 kernel for this policy.

Only shape/device/seed constants are cached. Mask features, row counts and
permutations are recomputed on every call. CPU paths support reference checks.
"""

from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch
import triton
import triton.language as tl

WAVE_SIZE = 132
SVD_ITERATIONS = 15
SVD_SEED = 42


def _check_mask(mask: torch.Tensor) -> None:
    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [heads, Q blocks, KV blocks]")
    if min(mask.shape) <= 0 or not mask.is_contiguous():
        raise ValueError("mask dimensions must be nonzero and storage contiguous")


@triton.jit
def _features_kernel(
    mask_ptr, features_ptr, nnz_ptr, columns: tl.constexpr, block: tl.constexpr, nnz_only: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    hit = tl.load(mask_ptr + row * columns + col, col < columns, other=0).to(tl.int32)
    count = tl.sum(hit, 0)
    tl.store(nnz_ptr + row, count.to(tl.int64))
    if not nnz_only:
        value = count - tl.cumsum(hit, 0) + hit
        tl.store(features_ptr + row * columns + col, value.to(tl.float32), col < columns)


def _features_and_nnz(mask: torch.Tensor, *, nnz_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    _check_mask(mask)
    if not mask.is_cuda:
        nnz = mask.sum(2)
        features = torch.empty(1) if nnz_only else mask.flip(2).cumsum(2, dtype=torch.float32).flip(2)
        return features, nnz
    heads, rows, columns = mask.shape
    nnz = torch.empty((heads, rows), device=mask.device, dtype=torch.int64)
    features = torch.empty((1,) if nnz_only else mask.shape, device=mask.device, dtype=torch.float32)
    _features_kernel[(heads * rows,)](
        mask, features, nnz, columns, triton.next_power_of_2(columns), nnz_only, num_warps=8
    )
    return features, nnz


def _orthonormalize(value: torch.Tensor) -> torch.Tensor:
    columns = []
    for index in range(value.shape[2]):
        column = value[:, :, index]
        for previous in columns:
            coefficient = (column * previous).sum(dim=1, keepdim=True)
            column = column - coefficient * previous
        column = column / column.norm(dim=1, keepdim=True).clamp(min=1e-8)
        columns.append(column)
    return torch.stack(columns, dim=2)


@lru_cache(maxsize=16)
def _initial_basis(heads: int, columns: int, device: str) -> torch.Tensor:
    return _orthonormalize(
        torch.randn(
            heads,
            columns,
            2,
            device=device,
            dtype=torch.float32,
            generator=torch.Generator(device=device).manual_seed(SVD_SEED),
        )
    )


def _rank2_scores(features: torch.Tensor) -> torch.Tensor:
    right = _initial_basis(features.shape[0], features.shape[2], str(features.device))
    transpose = features.transpose(1, 2)
    for _ in range(SVD_ITERATIONS):
        left = torch.bmm(features, right)
        left = _orthonormalize(left - left.mean(1, keepdim=True))
        right = _orthonormalize(torch.bmm(transpose, left))
    # Each RIB subset is centered next, including the root; avoid another pass.
    return torch.bmm(features, right)


def head_capacities(rows: int, head: int, wave_size: int = WAVE_SIZE) -> tuple[int, ...]:
    """Intersect one head with the nominal flattened launch buckets."""
    if rows <= 0 or head < 0 or wave_size <= 0:
        raise ValueError("rows/wave_size must be positive and head nonnegative")
    offset = head * rows % wave_size
    first = min(rows, wave_size - offset)
    left = rows - first
    return (first,) + (wave_size,) * (left // wave_size) + ((left % wave_size,) if left % wave_size else ())


class _Level(NamedTuple):
    head: torch.Tensor
    index: torch.Tensor
    valid: torch.Tensor
    selected: torch.Tensor
    target: torch.Tensor
    length: torch.Tensor


@lru_cache(maxsize=16)
def _layouts(heads: int, rows: int, device: str, wave_size: int) -> tuple[_Level, ...]:
    nodes = [(head, 0, head_capacities(rows, head, wave_size)) for head in range(heads)]
    levels = []
    while any(len(capacities) > 1 for _, _, capacities in nodes):
        active = [(head, start, capacities) for head, start, capacities in nodes if len(capacities) > 1]
        lengths = np.array([sum(capacities) for _, _, capacities in active])
        width = int(lengths.max())
        head = np.array([head for head, _, _ in active], dtype=np.int64)
        index = np.array([start + np.arange(width) for _, start, _ in active])
        valid = np.arange(width)[None] < lengths[:, None]
        selected = np.flatnonzero(valid.ravel())
        target = (head[:, None] * rows + index).ravel()[selected]
        levels.append(
            _Level(
                torch.as_tensor(head, device=device),
                torch.as_tensor(index.clip(0, rows - 1), device=device),
                torch.as_tensor(valid, device=device),
                torch.as_tensor(selected, device=device),
                torch.as_tensor(target, device=device),
                torch.as_tensor(lengths, device=device),
            )
        )
        next_nodes = []
        for head, start, capacities in nodes:
            if len(capacities) == 1:
                continue
            middle = len(capacities) // 2
            cut = sum(capacities[:middle])
            next_nodes.extend([(head, start, capacities[:middle]), (head, start + cut, capacities[middle:])])
        nodes = next_nodes
    return tuple(levels)


def _rib_order(points: torch.Tensor, wave_size: int = WAVE_SIZE) -> torch.Tensor:
    heads, rows, _ = points.shape
    order = torch.arange(rows, device=points.device)[None].expand(heads, -1).contiguous().clone()
    for layout in _layouts(heads, rows, str(points.device), wave_size):
        local = order[layout.head[:, None], layout.index]
        x = points[layout.head[:, None], local].double()
        valid = layout.valid[:, :, None]
        mean = (x * valid).sum(1) / layout.length[:, None]
        centered = (x - mean[:, None, :]) * valid
        covariance = torch.bmm(centered.transpose(1, 2), centered)
        _, vectors = torch.linalg.eigh(covariance)
        axis = vectors[:, :, -1]
        dominant = axis.abs().argmax(1, keepdim=True)
        sign = axis.gather(1, dominant).sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        axis = axis * sign
        key = (centered * axis[:, None, :]).sum(2).masked_fill(~layout.valid, torch.inf)
        local = local.gather(1, key.argsort(dim=1, stable=True))
        chosen = local.flatten().index_select(0, layout.selected)
        order = order.flatten().scatter(0, layout.target, chosen).reshape(heads, rows)
    return order


@torch.inference_mode()
def stage2_head_aligned(order: torch.Tensor, nnz: torch.Tensor, wave_size: int = WAVE_SIZE) -> torch.Tensor:
    """Stable NNZ order within the same head intersections used by RIB.

    ``order`` is a per-head bijection and ``nnz`` contains nonnegative counts.
    This does not move work between heads or globally sort a mixed-head bucket.
    """
    if order.ndim != 2 or order.shape != nnz.shape or order.device != nnz.device:
        raise ValueError("order and nnz must have matching [heads, rows] shape/device")
    heads, rows = order.shape
    if min(heads, rows, wave_size) <= 0 or order.dtype != torch.int64:
        raise ValueError("positive dimensions/wave_size and int64 order are required")
    offsets = torch.arange(heads, device=order.device) * rows % wave_size
    width = ((rows + 2 * wave_size - 2) // wave_size) * wave_size
    slots = offsets[:, None] + torch.arange(rows, device=order.device)[None]
    padded_order = torch.full((heads, width), -1, dtype=order.dtype, device=order.device)
    scores = torch.full((heads, width), -1, dtype=nnz.dtype, device=nnz.device)
    padded_order.scatter_(1, slots, order)
    scores.scatter_(1, slots, nnz.gather(1, order))
    ranking = scores.reshape(heads, -1, wave_size).argsort(dim=2, descending=True, stable=True)
    emitted = padded_order.reshape(heads, -1, wave_size).gather(2, ranking)
    return emitted[emitted >= 0].reshape(heads, rows).contiguous()


@torch.inference_mode()
def stage1(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native RIB order and fresh NNZ, before Stage2."""
    features, nnz = _features_and_nnz(mask)
    return _rib_order(_rank2_scores(features)), nnz


@torch.inference_mode()
def reorder(mask: torch.Tensor, *, stage2_only: bool = False) -> torch.Tensor:
    """Return one fresh [heads, rows] permutation for the existing attention path."""
    if stage2_only:
        _, nnz = _features_and_nnz(mask, nnz_only=True)
        order = torch.arange(mask.shape[1], device=mask.device)[None].expand(mask.shape[0], -1).contiguous()
    else:
        order, nnz = stage1(mask)
    return stage2_head_aligned(order, nnz)
