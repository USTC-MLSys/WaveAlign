import pickle
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import time
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
MASK_DUMP_IDX = 0  # 全局变量用于mask保存编号
from vllm.logger import init_logger
import atexit
from .greedy_reorder_cuda import global_greedy_reorder_cuda, global_banded_greedy_reorder_cuda, optimal_reorder_cuda, wave_overlap_reorder, nnz_3step_reorder, bcg_l2_reorder_cuda
from .reorder_rows_graph import reorder_rows_by_similarity
from flash_attn.cute.interface import flash_attn_func as flash_attn_func_for_sparse_reorder
from flash_attn.cute.interface import _flash_attn_fwd as _fa4_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
import os
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from einops import rearrange
# Lazy imports to avoid scipy/regex hang during multiprocessing spawn
# from spas_sage_attn.utils import get_block_map_meansim
# from vllm_omni.diffusion.attention.backends.autotune import SparseAttentionMeansim
_get_block_map_meansim = None
_SparseAttentionMeansim = None

def get_block_map_meansim(*args, **kwargs):
    global _get_block_map_meansim
    if _get_block_map_meansim is None:
        from spas_sage_attn.utils import get_block_map_meansim as _fn
        _get_block_map_meansim = _fn
    return _get_block_map_meansim(*args, **kwargs)

def SparseAttentionMeansim(*args, **kwargs):
    global _SparseAttentionMeansim
    if _SparseAttentionMeansim is None:
        from vllm_omni.diffusion.attention.backends.autotune import SparseAttentionMeansim as _cls
        _SparseAttentionMeansim = _cls
    return _SparseAttentionMeansim(*args, **kwargs)
logger = init_logger(__name__)
import vllm_omni.diffusion.models.wan2_2.global_step as global_step


def lexicographical_reorder(mask_float):
    Q, K = mask_float.shape
    device = mask_float.device
    col_density = mask_float.sum(dim=0)
    _, sorted_col_indices = torch.sort(col_density, descending=True)
    sorted_mask = mask_float[:, sorted_col_indices]
    num_ints = (K + 61) // 62
    keys = []
    pow2_full = 2 ** torch.arange(61, -1, -1, device=device, dtype=torch.float64)
    for i in range(num_ints - 1, -1, -1):
        start = i * 62
        end = min((i + 1) * 62, K)
        chunk = sorted_mask[:, start:end]
        p2 = pow2_full[62 - (end - start):] 
        sig = torch.mv(chunk.double(), p2).long()
        keys.append(-sig)
    sorted_row_indices = torch.arange(Q, device=device)
    for k in keys:
        k_perm = k[sorted_row_indices]
        _, sort_idx = torch.sort(k_perm, stable=True)
        sorted_row_indices = sorted_row_indices[sort_idx]
    return sorted_row_indices

def greedy_intersection_reorder_dense(mask_float):
    Q, K = mask_float.shape
    device = mask_float.device
    visited = torch.zeros(Q, dtype=torch.bool, device=device)
    permutation = [0]
    visited[0] = True
    current_idx = 0
    for _ in range(Q - 1):
        intersection = torch.mv(mask_float, mask_float[current_idx])
        intersection[visited] = -1.0
        next_idx = torch.argmax(intersection).item()
        permutation.append(next_idx)
        visited[next_idx] = True
        current_idx = next_idx
    return torch.tensor(permutation, device=device)

def batched_chunked_greedy_multihead(masks_float, chunk_size=32):
    """Batch greedy reorder across all heads at once.
    Args:
        masks_float: (H, MB, NB) — per-head block masks
    Returns:
        (H, MB) — per-head permutations
    """
    device = masks_float.device
    H, Q, K = masks_float.shape
    
    # Step 1: Lex reorder each head (fast GPU ops, hard to batch due to per-head col ordering)
    lex_perms = torch.zeros((H, Q), dtype=torch.long, device=device)
    for h in range(H):
        lex_perms[h] = lexicographical_reorder(masks_float[h])
    
    # Step 2: Apply lex perms to reorder masks — vectorized via gather
    # lex_perms: (H, Q) → gather index for dim=1 of masks_float (H, Q, K)
    new_masks = torch.gather(masks_float, 1, lex_perms.unsqueeze(-1).expand(H, Q, K))
    
    # Step 3: Chunk all heads together into one big batch
    B_per_head = Q // chunk_size
    if B_per_head == 0:
        chunk_size = Q
        B_per_head = 1
    cutoff = B_per_head * chunk_size
    
    # (H, cutoff, K) → (H * B_per_head, chunk_size, K)
    mask_chunked = new_masks[:, :cutoff, :].reshape(H * B_per_head, chunk_size, K)
    
    # Step 4: Batched greedy on ALL chunks across ALL heads simultaneously
    total_B = H * B_per_head
    S = torch.bmm(mask_chunked, mask_chunked.transpose(1, 2))  # (total_B, cs, cs)
    
    batch_idx = torch.arange(total_B, device=device)
    visited = torch.zeros((total_B, chunk_size), dtype=torch.bool, device=device)
    permutation = torch.zeros((total_B, chunk_size), dtype=torch.long, device=device)
    
    curr = torch.zeros(total_B, dtype=torch.long, device=device)
    permutation[:, 0] = curr
    visited[batch_idx, curr] = True
    
    for i in range(1, chunk_size):
        row = S[batch_idx, curr].clone()
        row[visited] = -1.0
        nxt = torch.argmax(row, dim=1)
        permutation[:, i] = nxt
        visited[batch_idx, nxt] = True
        curr = nxt
    
    # Step 5: Reconstruct per-head global permutations
    # (total_B, chunk_size) → (H, B_per_head, chunk_size)
    permutation = permutation.view(H, B_per_head, chunk_size)
    batch_offsets = torch.arange(0, cutoff, chunk_size, device=device).view(1, B_per_head, 1)
    flat_perm_local = (permutation + batch_offsets).view(H, -1)  # (H, cutoff)
    
    # Map back through lex perm: flat_perm_global[h, j] = lex_perms[h, flat_perm_local[h, j]]
    flat_perm_global = torch.gather(lex_perms, 1, flat_perm_local)  # (H, cutoff)
    
    if cutoff < Q:
        # Handle remainder for each head (typically very small)
        rem_perms = []
        for h in range(H):
            rem_perm = greedy_intersection_reorder_dense(new_masks[h, cutoff:])
            rem_global = lex_perms[h, cutoff + rem_perm]
            rem_perms.append(rem_global)
        rem_all = torch.stack(rem_perms)  # (H, Q - cutoff)
        final_perm = torch.cat([flat_perm_global, rem_all], dim=1)
    else:
        final_perm = flat_perm_global
    
    return final_perm

# ──────────────────────────────────────────────────────────────────────────────
# LongCat BSA helpers — strictly following the original LongCat-Video repo
# ──────────────────────────────────────────────────────────────────────────────

def _longcat_rearrange_THW_to_3d_block(x, Nt, Nh, Nw, t, h, w):
    """Rearrange from THW order to 3D block order.
    
    Input x:  (B, H, Nt*t * Nh*h * Nw*w, D)
    Output:   (B, H, Nt*Nh*Nw * t*h*w, D)
    """
    B, H_heads, _, D = x.shape
    x = x.view(B, H_heads, Nt, t, Nh, h, Nw, w, D)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8)  # B H Nt Nh Nw t h w D
    return x.contiguous().view(B, H_heads, Nt * Nh * Nw * t * h * w, D)


def _longcat_rearrange_3d_block_to_THW(x, Nt, Nh, Nw, t, h, w):
    """Rearrange from 3D block order back to THW order.
    
    Input x:  (B, H, Nt*Nh*Nw * t*h*w, D)
    Output:   (B, H, Nt*t * Nh*h * Nw*w, D)
    """
    B, H_heads, _, D = x.shape
    x = x.view(B, H_heads, Nt, Nh, Nw, t, h, w, D)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7, 8)  # B H Nt t Nh h Nw w D
    return x.contiguous().view(B, H_heads, Nt * t * Nh * h * Nw * w, D)


def _longcat_mean_pooling_compression(x, block_size):
    """Mean-pool along the sequence dimension in blocks.
    
    Input x:  (B, H, S, D)
    Output:   (B, H, num_blocks, D)
    """
    import math as _math
    B, H, S, D = x.shape
    num_blocks = _math.ceil(S / block_size)
    if S % block_size != 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, num_blocks * block_size - S))
    return x.view(B, H, num_blocks, block_size, D).float().mean(dim=3).to(x.dtype)


def _longcat_get_select_indices_cdf(score, cdf_threshold, sm_scale):
    """CDF-based block selection — exact copy of LongCat's get_select_indices_cdf_from_score.
    
    Args:
        score: (B, H, num_q_blocks, num_k_blocks) — raw QK^T block scores
        cdf_threshold: float — cumulative probability cutoff
        sm_scale: float — softmax temperature (typically 1/sqrt(D))
    
    Returns:
        block_indices: (B, H, num_q_blocks, num_k_blocks) — sorted column indices
        block_indices_lens: (B, H, num_q_blocks) — number of selected blocks per row
    """
    weights = torch.softmax(score * sm_scale, dim=-1)
    B, H, Sq, Sk = weights.shape
    thr = torch.full((H,), cdf_threshold, device=weights.device).view(1, H, 1, 1).expand(B, -1, Sq, -1)
    weights_sorted = torch.sort(weights, dim=-1, descending=True)
    cdf = torch.cumsum(weights_sorted.values, dim=-1)
    num_selected = torch.searchsorted(cdf, thr, right=True)
    return weights_sorted.indices, num_selected.squeeze(-1)


def _longcat_get_select_indices_topk(score, sparsity):
    """Top-k block selection — exact copy of LongCat's get_select_indices_topk_from_score.
    
    Args:
        score: (B, H, num_q_blocks, num_k_blocks)
        sparsity: float — fraction of blocks to DROP (e.g. 0.875 keeps 12.5%)
    
    Returns:
        block_indices: (B, H, num_q_blocks, num_selected)
        block_indices_lens: (B, H, num_q_blocks) — all equal to num_selected
    """
    num_selected = int((1 - sparsity) * score.shape[-1])
    block_indices = torch.topk(score, num_selected)[1]
    block_indices_lens = torch.full(
        (block_indices.shape[0], block_indices.shape[1], block_indices.shape[2]),
        num_selected,
        dtype=torch.int32,
        device=block_indices.device,
    )
    return block_indices, block_indices_lens


def _longcat_get_select_indices_cdf_topk(score, sparsity, cdf_threshold, sm_scale):
    """CDF + top-k block selection — exact copy of LongCat's get_select_indices_cdf_topk_from_score.
    
    Selects max(cdf_count, topk_count) blocks per query block.
    """
    weights = torch.softmax(score * sm_scale, dim=-1)
    B, H, Sq, Sk = weights.shape
    thr = torch.full((H,), cdf_threshold, device=weights.device).view(1, H, 1, 1).expand(B, -1, Sq, -1)
    weights_sorted = torch.sort(weights, dim=-1, descending=True)
    cdf = torch.cumsum(weights_sorted.values, dim=-1)
    num_selected = torch.searchsorted(cdf, thr, right=True)
    # max(cdf, topk)
    num_selected_topk = int((1 - sparsity) * score.shape[-1])
    num_selected[num_selected < num_selected_topk] = num_selected_topk
    return weights_sorted.indices, num_selected.squeeze(-1)


def _longcat_indices_to_fa4_blocksparse(block_indices, block_indices_lens, num_k_blocks, batch_size):
    """Convert LongCat's (block_indices, block_indices_lens) to FA4 BlockSparseTensorsTorch format.
    
    Args:
        block_indices: (B, H, num_q_blocks, max_k_blocks) — sorted column indices per Q block
        block_indices_lens: (B, H, num_q_blocks) — valid count per Q block row
        num_k_blocks: int — total number of K blocks (NB)
        batch_size: int — batch size for the FA4 call (may differ from B if B=1)
    
    Returns:
        cnt, idx tensors for BlockSparseTensorsTorch
    """
    B_mask, H, MB, max_k = block_indices.shape
    
    cnt = block_indices_lens.to(torch.int32)
    if cnt.shape[0] == 1 and batch_size > 1:
        cnt = cnt.expand(batch_size, -1, -1).contiguous()
    
    idx = block_indices.to(torch.int32)
    if max_k < num_k_blocks:
        idx = torch.nn.functional.pad(idx, (0, num_k_blocks - max_k), value=0)
    elif max_k > num_k_blocks:
        idx = idx[..., :num_k_blocks]
    
    col_range = torch.arange(idx.shape[-1], device=idx.device).view(1, 1, 1, -1)
    valid = col_range < cnt.unsqueeze(-1)
    sort_perm = torch.argsort(~valid, dim=-1, stable=True).int()
    idx = torch.gather(idx, -1, sort_perm.long())
    idx = idx * valid.int()
    
    if idx.shape[0] == 1 and batch_size > 1:
        idx = idx.expand(batch_size, -1, -1, -1).contiguous()
    
    return cnt, idx

class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl




class FlashAttentionImpl(AttentionImpl):
    default_state_dir = '/data1/dhelix_shared/daizijian/SpargeAttn/evaluate/models_dict/result'
    num_layers = 80
    base_num_layers = 40
    autotune_params = [None] * base_num_layers
    loaded_param_path = None
    # 全局 layer 索引和参数缓存
    layeridx = 0
    total_sparse_ms = 0.0
    total_reorder_ms = 0.0
    total_attn_ms = 0.0
    total_o_shuffle_ms = 0.0
    total_r_algo_ms = 0.0
    total_r_qgather_ms = 0.0
    total_r_sparse_ms = 0.0
    total_r_expand_ms = 0.0
    count = 0
    # MFU tracking: accumulate actual sparse FLOPs and dense-equivalent FLOPs
    total_attn_flops = 0.0      # actual sparse FLOPs (density-weighted)
    total_dense_flops = 0.0     # dense-equivalent FLOPs (for reference)
    sum_density = 0.0           # sum of per-call avg density
    density_count = 0           # number of density observations
    H100_BF16_PEAK_TFLOPS = 989.5  # H100 SXM BF16 peak

    @staticmethod
    def _autotune_print_level() -> int:
        # 0: only critical errors, 1: summary, 2: per-head, 3: verbose diagnostics
        try:
            return int(os.environ.get("SPARGE_AUTOTUNE_PRINT_LEVEL", "1"))
        except Exception:
            return 1

    @classmethod
    def _autotune_print(cls, msg: str, level: int = 1, rank0_only: bool = False) -> None:
        if cls._autotune_print_level() < level:
            return
        if rank0_only and os.environ.get("RANK", "0") != "0":
            return
        print(msg, flush=True)

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        # 每个实例持有自己的tuner对象
        self.sparse_attn_tuner = SparseAttentionMeansim()

    @staticmethod
    def _unwrap_flash_output(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        # FA3 may return (out, lse), FA2 returns out
        return out[0] if isinstance(out, tuple) else out

    @classmethod
    def _resolve_param_path(cls) -> str:
        state_path = os.environ.get("SPARGE_STATE_PATH")
        if state_path:
            base_path = state_path
        else:
            frame = os.environ.get("SPARGE_FRAME")
            if not frame:
                raise RuntimeError(
                    "Missing SPARGE_FRAME/SPARGE_STATE_PATH: legacy global autotune_params.pkl is disabled."
                )
            base_dir = os.environ.get("SPARGE_STATE_DIR", cls.default_state_dir)
            base_path = os.path.join(base_dir, f"wan_saved_state_dict_frame{frame}.pt")

        # Keep rank-specific state files to avoid TP ranks overwriting each other.
        rank = os.environ.get("RANK", "0")
        root, ext = os.path.splitext(base_path)
        if ext:
            return f"{root}.rank{rank}{ext}"
        return f"{base_path}.rank{rank}"

    @classmethod
    def _serialize_params(cls, params: list[dict | None]) -> dict:
        return {
            "format": "vllm_sparge_autotune_v1",
            "base_num_layers": cls.base_num_layers,
            "num_layers": cls.num_layers,
            "autotune_params": params,
        }

    @classmethod
    def _normalize_loaded_params(cls, loaded_obj) -> list:
        merged_params = [None] * cls.base_num_layers
        if isinstance(loaded_obj, dict):
            loaded_params = loaded_obj.get("autotune_params")
            if loaded_params is None and all(str(i) in loaded_obj for i in range(cls.base_num_layers)):
                loaded_params = [loaded_obj.get(str(i)) for i in range(cls.base_num_layers)]
        else:
            loaded_params = loaded_obj

        if isinstance(loaded_params, list):
            for i, value in enumerate(loaded_params[:cls.base_num_layers]):
                if value is not None:
                    merged_params[i] = value
        return merged_params

    @classmethod
    def _read_saved_params(cls, param_path: str):
        if param_path.endswith((".pt", ".pth")):
            try:
                return torch.load(param_path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(param_path, map_location="cpu")
        with open(param_path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def _write_saved_params(cls, param_path: str, payload) -> None:
        if param_path.endswith((".pt", ".pth")):
            torch.save(payload, param_path)
            return
        with open(param_path, 'wb') as f:
            pickle.dump(payload, f)

    @classmethod
    def _init_param_storage(cls) -> None:
        if cls.autotune_params is None or len(cls.autotune_params) != cls.base_num_layers:
            cls.autotune_params = [None] * cls.base_num_layers

    @classmethod
    def _get_layer_param(cls, layeridx: int):
        cls._init_param_storage()
        return cls.autotune_params[layeridx]

    @classmethod
    def _set_layer_param(cls, layeridx: int, value) -> None:
        cls._init_param_storage()
        cls.autotune_params[layeridx] = value

    @classmethod
    def _merge_and_save_params(cls) -> None:
        param_path = cls._resolve_param_path()
        param_dir = os.path.dirname(param_path)
        if param_dir:
            os.makedirs(param_dir, exist_ok=True)
        merged_params = [None] * cls.base_num_layers
        if os.path.exists(param_path):
            try:
                old_params = cls._read_saved_params(param_path)
                merged_params = cls._normalize_loaded_params(old_params)
            except Exception as e:
                print(f"[autotune] Read params failed: {e}")
        for i, value in enumerate(cls.autotune_params):
            if value is not None:
                merged_params[i] = value
        payload = cls._serialize_params(merged_params) if param_path.endswith((".pt", ".pth")) else merged_params
        cls._write_saved_params(param_path, payload)
        cls.autotune_params = merged_params
        cls.loaded_param_path = param_path
        print(f"[autotune] Saved merged autotune_params to {param_path}")

    @classmethod
    def _save_params_for_rank(cls) -> None:
        try:
            cls._merge_and_save_params()
        except Exception as e:
            print(f"[autotune] Save params failed: {e}")

    @classmethod
    def _load_params_from_disk_if_needed(cls) -> None:
        param_path = cls._resolve_param_path()
        if cls.loaded_param_path == param_path and cls.autotune_params is not None:
            return
        cls._init_param_storage()
        if os.path.exists(param_path):
            try:
                loaded_params = cls._read_saved_params(param_path)
                cls.autotune_params = cls._normalize_loaded_params(loaded_params)
                cls._autotune_print(f"[autotune] Loaded Sparge state from {param_path}", level=1)
            except Exception as e:
                print(f"[autotune] Load params failed: {e}")
                cls.autotune_params = [None] * cls.base_num_layers
        else:
            cls.autotune_params = [None] * cls.base_num_layers
        cls.loaded_param_path = param_path

    @staticmethod
    def _get_global_step() -> int:
        try:
            return int(getattr(global_step, "global_step", -1))
        except Exception:
            return -1

    def _forward_varlen_masked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            _pad_input,
            _unpad_input,
            _upad_input,
            flash_attn_varlen_func,
        )

        assert attention_mask.ndim == 2, "attention_mask must be 2D, (batch_size, seq_len)"
        query_length = query.size(1)
        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
            query, key, value, attention_mask, query_length, _unpad_input
        )

        out_unpad = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            **{
                "causal": self.causal,
                "softmax_scale": self.softmax_scale,
            },
        )
        out_unpad = self._unwrap_flash_output(out_unpad)
        return _pad_input(out_unpad, indices_q, query.size(0), query_length)

    def _forward_varlen_dense(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Common wrapper for calling flash_attn_varlen_func for XPU and CUDA in vLLM.

        NOTE: careful to keep the kwargs on these aligned and to pass everything as a keyword
        argument, because some of the args differ positionally at the moment.

        https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/vllm_flash_attn/flash_attn_interface.py#L176
        https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/_xpu_ops.py#L310
        """
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            flash_attn_varlen_func,
        )

        batch_size, q_len = query.size()[:2]
        cu_seqlens = torch.arange(0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
        # b s ... -> (b s) ...
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        out = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=q_len,
            max_seqlen_k=q_len,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        out = self._unwrap_flash_output(out)
        # (b s) h d -> b s h d
        return out.reshape(batch_size, q_len, *out.shape[1:])

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """CUDA/ROCm/MUSA flash attention implementation."""
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            HAS_FLASH_ATTN,
        )

        if not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionBackend requires Flash Attention. "
                "Please install one of: fa3-fwd, flash-attention, or flash-attn. "
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        def _dense_attention_forward(q, k, v):
            attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None
            if attention_mask is not None and torch.any(~attention_mask):
                return self._forward_varlen_masked(q, k, v, attention_mask)
            dense_out = flash_attn_func(
                q,
                k,
                v,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
            )
            return self._unwrap_flash_output(dense_out)

        current_global_step = FlashAttentionImpl._get_global_step()
        tune_mode = int(os.environ.get("TUNE_MODE", 0))
        sparse_warmup_steps = int(os.environ.get("SPARSE_WARMUP_STEPS", 0))
        sparse_ready = (
            tune_mode == 1
            or current_global_step < 0
            or current_global_step >= sparse_warmup_steps
        )

        if key.shape[1] > 4096 and sparse_ready:
            # print(f"sparse attn {current_global_step=} {FlashAttentionImpl.layeridx=}")
            sparse_mode = os.environ.get("SPARSE_MODE", "xattn")
            threshold = float(os.environ.get("VLLM_THRESHOLD", 0.9))
            mode = os.environ.get("VLLM_MODE", "nnz_reorder")

            # ── Dense baseline: bypass all sparse logic, measure pure dense FA MFU ──
            if sparse_mode == "dense":
                batch_size, L, H, D = query.shape
                sd1 = torch.cuda.Event(enable_timing=True)
                sd2 = torch.cuda.Event(enable_timing=True)
                sd1.record()
                dense_out = flash_attn_func(
                    query, key, value,
                    causal=self.causal,
                    softmax_scale=self.softmax_scale,
                )
                dense_out = self._unwrap_flash_output(dense_out)
                sd2.record()
                torch.cuda.synchronize()
                attn_ms = sd1.elapsed_time(sd2)
                FlashAttentionImpl.total_attn_ms += attn_ms
                # Dense FLOPs with density=1.0
                dense_flops = 4.0 * H * L * L * D
                FlashAttentionImpl.total_attn_flops += dense_flops
                FlashAttentionImpl.total_dense_flops += dense_flops
                FlashAttentionImpl.sum_density += 1.0
                FlashAttentionImpl.density_count += 1
                FlashAttentionImpl.count += 1
                FlashAttentionImpl.layeridx += 1
                if FlashAttentionImpl.layeridx >= FlashAttentionImpl.num_layers:
                    FlashAttentionImpl.layeridx = 0
                print(f"[Dense] L={L} H={H} D={D} attn={attn_ms:.2f}ms")
                return dense_out
            layeridx = FlashAttentionImpl.layeridx
            logical_layeridx = layeridx % FlashAttentionImpl.base_num_layers
            # autotune模式：调优并保存参数到磁盘
            # 手动设置本次 autotune 的 layeridx 范围（每次手动修改即可）
            LAYERIDX_START = 0   # 包含
            LAYERIDX_END = 33  # 不包含，例：20-40 表示第20到39层
            if tune_mode == 1:
                if (LAYERIDX_START <= layeridx < LAYERIDX_END):
                    # 初始化参数缓存
                    FlashAttentionImpl._init_param_storage()
                    if FlashAttentionImpl._get_layer_param(logical_layeridx) is None:
                        
                        q_sample = rearrange(query, '... L H D -> ... H L D')
                        k_sample = rearrange(key, '... L H D -> ... H L D')
                        v_sample = rearrange(value, '... L H D -> ... H L D')
                        # print(f"{q_sample.shape=} {k_sample.shape=} {v_sample.shape=}",flush=True)
                        tune_pv = int(os.environ.get("SPARGE_TUNE_PV", "0")) != 0
                        tuner = SparseAttentionMeansim(tune_pv=tune_pv)
                        FlashAttentionImpl._autotune_print(
                            f"[autotune] layer={layeridx} logical_layer={logical_layeridx} "
                            f"SPARGE_TUNE_PV={int(tune_pv)}",
                            level=1,
                        )
                        tuner.init_hyperparams(self.num_heads, q_sample.device)
                        mean_sparsities = []
                        head_time_ms = []
                        for head_idx in range(self.num_heads):
                            t_head_start = time.perf_counter()
                            FlashAttentionImpl._autotune_print(
                                f"[autotune] layer={layeridx} logical_layer={logical_layeridx} "
                                f"head={head_idx + 1}/{self.num_heads} start",
                                level=2,
                            )
                            # 判断条件：layeridx==33, head_idx==16, rank==1
                            rank_env = int(os.environ.get("RANK", "0"))
                            if (layeridx == 33 and head_idx in [10,16,19] and rank_env == 1) or (layeridx == 34 and head_idx in [13] and rank_env == 1) or (layeridx == 38 and head_idx in [6,13] and rank_env == 1):
                                # 从 481 的对应文件读取参数
                                param_path_521 = "/data1/dhelix_shared/daizijian/SpargeAttn/evaluate/models_dict/result/wan_saved_state_dict_frame521.rank1.pt"
                                try:
                                    loaded_param = torch.load(param_path_521)
                                    # 兼容dict结构
                                    if isinstance(loaded_param, dict) and "autotune_params" in loaded_param:
                                        params_list = loaded_param["autotune_params"]
                                    elif isinstance(loaded_param, list):
                                        params_list = loaded_param
                                    else:
                                        raise RuntimeError("Unknown param file structure for Sparge state.")
                                    param = params_list[layeridx]
                                    # param 应为 dict，包含 cdfthreshd, simthreshd1, pvthreshd
                                    # 取出每个参数的第16个 head
                                    cdfthreshd = param['cdfthreshd'][head_idx].clone() if 'cdfthreshd' in param else None
                                    simthreshd1 = param['simthreshd1'][head_idx].clone() if 'simthreshd1' in param else None
                                    pvthreshd = param['pvthreshd'][head_idx].clone() if 'pvthreshd' in param else None
                                    # 用于调试输出
                                    FlashAttentionImpl._autotune_print(
                                        f"[autotune][special] layer={layeridx} head={head_idx} loaded cdfthreshd={cdfthreshd} simthreshd1={simthreshd1} pvthreshd={pvthreshd}",
                                        level=1,
                                    )
                                    # 用这些参数替换当前tuner的对应head参数
                                    if cdfthreshd is not None:
                                        tuner.cdfthreshd[head_idx] = cdfthreshd
                                    if simthreshd1 is not None:
                                        tuner.simthreshd1[head_idx] = simthreshd1
                                    if pvthreshd is not None:
                                        tuner.pvthreshd[head_idx] = pvthreshd
                                    # 计算 mean_sparsity（可选，或直接设为0）
                                    rtdict = {'mean_sparsity': 0.0}
                                except Exception as e:
                                    FlashAttentionImpl._autotune_print(
                                        f"[autotune][special][ERROR] Failed to load param from {param_path_521}: {e}",
                                        level=1,
                                    )
                            else:
                                rtdict = tuner.autotune(
                                    q_sample[:, head_idx:head_idx+1],
                                    k_sample[:, head_idx:head_idx+1],
                                    v_sample[:, head_idx:head_idx+1],
                                    head_idx=head_idx
                                )
                                tuner.fill_results(rtdict)
                            mean_sparsities.append(rtdict['mean_sparsity'])
                            elapsed_ms = (time.perf_counter() - t_head_start) * 1000.0
                            head_time_ms.append(elapsed_ms)
                            avg_ms = sum(head_time_ms) / len(head_time_ms)
                            FlashAttentionImpl._autotune_print(
                                f"[autotune] layer={layeridx} logical_layer={logical_layeridx} "
                                f"head={head_idx + 1}/{self.num_heads} done "
                                f"mean_sparsity={rtdict['mean_sparsity']:.6f} "
                                f"head_time_ms={elapsed_ms:.2f} avg_head_time_ms={avg_ms:.2f}",
                                level=1,
                            )
                            # --- 分布式同步，每个head结束后barrier ---
                            try:
                                if torch.distributed.is_initialized():
                                    torch.distributed.barrier()
                                    FlashAttentionImpl._autotune_print(
                                        f"[autotune][barrier] layer={layeridx} logical_layer={logical_layeridx} head={head_idx + 1}/{self.num_heads} barrier同步完成 (rank={os.environ.get('RANK','?')})",
                                        level=1,
                                        rank0_only=False,
                                    )
                            except Exception as e:
                                FlashAttentionImpl._autotune_print(
                                    f"[autotune][barrier][ERROR] layer={layeridx} logical_layer={logical_layeridx} head={head_idx + 1}/{self.num_heads} barrier异常: {e}",
                                    level=1,
                                    rank0_only=False,
                                )
                        if len(mean_sparsities) > 0:
                            avg_sparsity = float(sum(mean_sparsities)) / len(mean_sparsities)
                            FlashAttentionImpl._autotune_print(
                                f"[autotune] mean_sparsity across all heads: {avg_sparsity:.6f}",
                                level=1,
                            )
                        if layeridx == 33 and rank_env == 1:
                            print(f"jjjjj cdfthreshd={tuner.cdfthreshd.detach().cpu()} simthreshd1={tuner.simthreshd1.detach().cpu()} pvthreshd={tuner.pvthreshd.detach().cpu()}")
                        FlashAttentionImpl._set_layer_param(logical_layeridx, {
                            'cdfthreshd': tuner.cdfthreshd.detach().cpu(),
                            'simthreshd1': tuner.simthreshd1.detach().cpu(),
                            'pvthreshd': tuner.pvthreshd.detach().cpu(),
                        })
                    # Save once after finishing a full logical block (0..39) to reduce IO churn.
                    if logical_layeridx == LAYERIDX_END - 1:
                        FlashAttentionImpl._save_params_for_rank()
                    # 递增layeridx
                    FlashAttentionImpl.layeridx += 1
                    if FlashAttentionImpl.layeridx >= FlashAttentionImpl.num_layers:
                        FlashAttentionImpl.layeridx = 0
                    # Keep forward activations valid during tuning; returning zeros here
                    # can corrupt later-layer statistics and bias autotune sparsity.
                    return _dense_attention_forward(query, key, value)
                else:
                    # print(f"[autotune] else {layeridx=}")
                    FlashAttentionImpl.layeridx += 1
                    if FlashAttentionImpl.layeridx >= FlashAttentionImpl.num_layers:
                        FlashAttentionImpl.layeridx = 0
                    return _dense_attention_forward(query, key, value)
            # 非tune模式：只在首次需要时从磁盘读取参数
            if sparse_mode == "sparge":
                FlashAttentionImpl._autotune_print(
                    f"read sparge params for layer {logical_layeridx} from disk",
                    level=2,
                )
                FlashAttentionImpl._load_params_from_disk_if_needed()
                param = FlashAttentionImpl._get_layer_param(logical_layeridx)
                if param is not None:
                    # 兼容 tensor parallel 环境
                    cdfthreshd_full = param['cdfthreshd'].to(query.device)
                    simthreshd1_full = param['simthreshd1'].to(query.device)
                    pvthreshd = param['pvthreshd'].to(query.device)
                    tp_size = get_tensor_model_parallel_world_size() if 'get_tensor_model_parallel_world_size' in globals() else 1
                    tp_rank = get_tensor_model_parallel_rank() if 'get_tensor_model_parallel_rank' in globals() else 0
                    if cdfthreshd_full.numel() > 0 and simthreshd1_full.numel() > 0 and tp_size > 1 and cdfthreshd_full.numel() > query.shape[2]:
                        # 假设 head 数能整除 tp_size
                        num_heads = cdfthreshd_full.shape[0]
                        heads_per_rank = num_heads // tp_size
                        start = tp_rank * heads_per_rank
                        end = (tp_rank + 1) * heads_per_rank
                        cdfthreshd = cdfthreshd_full[start:end]
                        simthreshd1 = simthreshd1_full[start:end]
                    else:
                        cdfthreshd = cdfthreshd_full
                        simthreshd1 = simthreshd1_full
                else:
                    cdfthreshd = None
                    simthreshd1 = None
                    pvthreshd = None
                # 递增layeridx
                FlashAttentionImpl.layeridx += 1
                if FlashAttentionImpl.layeridx >= FlashAttentionImpl.num_layers:
                    FlashAttentionImpl.layeridx = 0

            # import os
            s1 = torch.cuda.Event(enable_timing=True)
            s2 = torch.cuda.Event(enable_timing=True)
            s3 = torch.cuda.Event(enable_timing=True)
            s4 = torch.cuda.Event(enable_timing=True)
            s5 = torch.cuda.Event(enable_timing=True)
            # threshold = float(os.environ.get("VLLM_THRESHOLD", 0.9))
            # mode = os.environ.get("VLLM_MODE", "nnz_reorder")
            # sparse_mode = os.environ.get("SPARSE_MODE", "xattn")
            s1.record()
            target_density = 0.1
            batch_size = query.shape[0]
            L = query.shape[1]
            H = query.shape[2]
            D = query.shape[3]

            # 3. Apply Reordering (Batched Chunked Greedy)
            # BLK: 64 for vsa mode (finer blocks), 128 for xattn/sparge (FA4 high-level default)
            BLK = int(os.environ.get("VLLM_BLK", 64 if sparse_mode == "vsa" else 128))
            
            L_padded = ((L + BLK - 1) // BLK) * BLK
            pad_len = L_padded - L
            
            if pad_len > 0:
                query = torch.nn.functional.pad(query, (0,0, 0,0, 0,pad_len))
                key = torch.nn.functional.pad(key, (0,0, 0,0, 0,pad_len))
                value = torch.nn.functional.pad(value, (0,0, 0,0, 0,pad_len))
                
            MB = L_padded // BLK
            NB = L_padded // BLK
            if FlashAttentionImpl.count == 0:
                FlashAttentionImpl._autotune_print(
                    f"[DIAG] L={L} L_padded={L_padded} MB={MB} NB={NB} H={H} D={D} batch={batch_size} mode={mode} thr={threshold}",
                    level=3,
                )
            
            # Estimate block attention via x-attention — PER HEAD
            from xattn.src.Xattention import xattn_estimate
            # xattn_estimate expects (B, num_heads, seq_len, head_dim)
            q_est = query.transpose(1, 2).contiguous()  # (B, H, L_padded, D)
            k_est = key.transpose(1, 2).contiguous()
            
            # Run xattn_estimate on ALL heads independently
            # mask_all_heads: (H, MB, NB) — per-head block mask
            mask_all_heads = torch.zeros((H, MB, NB), dtype=torch.int32, device='cuda')
            
            if sparse_mode == "xattn":
                if threshold is not None:
                    # Threshold-based logic — run on all heads at once
                    attn_sums, simple_masks = xattn_estimate(
                        q_est, k_est,
                        block_size=BLK, stride=BLK//4, norm=1,
                        threshold=threshold, select_mode="inverse",
                        use_triton=True, causal=False, 
                        chunk_size=4096, kdb=1
                    )
                    # simple_masks shape: (B, H, MB_est, NB_est)
                    MB_est, NB_est = simple_masks.shape[-2], simple_masks.shape[-1]
                    for h in range(H):
                        mask_all_heads[h, :min(MB, MB_est), :min(NB, NB_est)] = simple_masks[0, h, :min(MB, MB_est), :min(NB, NB_est)].int()
                else:
                    # Density-based Top-K logic — run on all heads at once
                    attn_sums, _ = xattn_estimate(
                        q_est, k_est,
                        block_size=BLK, stride=BLK//4, norm=1,
                        threshold=0.8, select_mode="inverse",
                        use_triton=True, causal=False, 
                        chunk_size=4096, kdb=1
                    )
                    # attn_sums shape: (B, H, MB_est, NB_est)
                    MB_est, NB_est = attn_sums.shape[-2], attn_sums.shape[-1]
                    
                    num_to_keep = int(MB * NB * target_density)
                    for h in range(H):
                        base_attn_sums = torch.zeros((MB, NB), device='cuda')
                        base_attn_sums[:min(MB, MB_est), :min(NB, NB_est)] = attn_sums[0, h, :min(MB, MB_est), :min(NB, NB_est)]
                        
                        attn_flat = base_attn_sums.view(-1).clone()
                        _, indices_flat = torch.topk(attn_flat, min(num_to_keep, (attn_flat > -1e9).sum().item()))
                        
                        mask_h = torch.zeros((MB, NB), dtype=torch.int32, device='cuda')
                        mask_h.view(-1)[indices_flat] = 1
                        mask_all_heads[h] = mask_h
            elif sparse_mode == "vsa":
                # VSA-style: block mean-pool → QK matmul → topk
                q_blocked = q_est.view(batch_size, H, MB, BLK, D)
                k_blocked = k_est.view(batch_size, H, NB, BLK, D)
                q_c = q_blocked.float().mean(dim=3).to(q_est.dtype)
                k_c = k_blocked.float().mean(dim=3).to(k_est.dtype)
                scores = torch.matmul(q_c, k_c.transpose(-1, -2)) / (D ** 0.5)  # (B, H, MB, NB)
                topk_n = max(1, int(NB * (1 - threshold)))
                topk_idx = scores[0].topk(topk_n, dim=-1).indices  # (H, MB, topk_n)
                mask_all_heads = torch.zeros((H, MB, NB), dtype=torch.int32, device='cuda')
                mask_all_heads.scatter_(-1, topk_idx, 1)
            elif sparse_mode == "longcat_cdf":
                # ──────────────────────────────────────────────────────────
                # LongCat BSA — strictly following the original LongCat-Video
                # repo. Self-contained path: 3D rearrange → gating → CDF/topk
                # selection → FA4 blocksparse → rearrange back.
                # No row reorder, no Q reorder.
                # ──────────────────────────────────────────────────────────
                batch_size_lc, L_orig, H_lc, D_lc = query.shape

                # --- 1. Read latent_shape from global variable ---
                lshape = getattr(global_step, 'latent_shape', None)
                if lshape is None:
                    raise RuntimeError(
                        "[longcat_cdf] global_step.latent_shape is None. "
                        "Make sure WanTransformer3DModel.forward() sets it."
                    )
                N_t, N_h, N_w = lshape
                assert N_t * N_h * N_w == L_orig, (
                    f"[longcat_cdf] latent_shape {lshape} product {N_t*N_h*N_w} != L={L_orig}"
                )

                # --- 2. Hardcoded chunk_3d_shape (matching original LongCat default) ---
                tq, hq, wq = 4, 4, 8   # chunk_3d_shape_q
                tk, hk, wk = 4, 4, 8   # chunk_3d_shape_k
                chunk_size_q = tq * hq * wq   # 128
                chunk_size_k = tk * hk * wk   # 128

                # Validate divisibility
                assert N_t % tq == 0 and N_h % hq == 0 and N_w % wq == 0, (
                    f"[longcat_cdf] latent_shape {lshape} not divisible by chunk_3d ({tq},{hq},{wq})"
                )
                assert N_t % tk == 0 and N_h % hk == 0 and N_w % wk == 0, (
                    f"[longcat_cdf] latent_shape {lshape} not divisible by chunk_3d ({tk},{hk},{wk})"
                )

                Ntq, Nhq, Nwq = N_t // tq, N_h // hq, N_w // wq
                Ntk, Nhk, Nwk = N_t // tk, N_h // hk, N_w // wk

                # --- 3. Transpose to (B, H, L, D) and do 3D block rearrange ---
                q_bhld = query.transpose(1, 2).contiguous()   # (B, H, L, D)
                k_bhld = key.transpose(1, 2).contiguous()
                v_bhld = value.transpose(1, 2).contiguous()

                q_3d = _longcat_rearrange_THW_to_3d_block(q_bhld, Ntq, Nhq, Nwq, tq, hq, wq)
                k_3d = _longcat_rearrange_THW_to_3d_block(k_bhld, Ntk, Nhk, Nwk, tk, hk, wk)
                v_3d = _longcat_rearrange_THW_to_3d_block(v_bhld, Ntk, Nhk, Nwk, tk, hk, wk)

                # --- 4. Gating: mean pooling compression ---
                q_cmp = _longcat_mean_pooling_compression(q_3d, chunk_size_q)  # (B, H, num_q_blocks, D)
                k_cmp = _longcat_mean_pooling_compression(k_3d, chunk_size_k)  # (B, H, num_k_blocks, D)

                # Block-level QK score
                score = torch.matmul(q_cmp, k_cmp.transpose(-1, -2))  # (B, H, MB, NB)
                sm_scale = 1.0 / (D_lc ** 0.5)

                # --- 5. Block selection ---
                longcat_sparsity_str = os.environ.get("LONGCAT_SPARSITY", None)
                longcat_sparsity = float(longcat_sparsity_str) if longcat_sparsity_str is not None else None

                if longcat_sparsity is not None and threshold is not None:
                    # CDF + topk mode
                    block_indices, block_indices_lens = _longcat_get_select_indices_cdf_topk(
                        score, longcat_sparsity, threshold, sm_scale
                    )
                elif longcat_sparsity is not None:
                    # topk-only mode
                    block_indices, block_indices_lens = _longcat_get_select_indices_topk(
                        score, longcat_sparsity
                    )
                else:
                    # CDF-only mode (default, threshold from VLLM_THRESHOLD)
                    block_indices, block_indices_lens = _longcat_get_select_indices_cdf(
                        score, threshold, sm_scale
                    )

                # --- 6. Print density stats ---
                num_q_blocks = q_cmp.shape[2]
                num_k_blocks = k_cmp.shape[2]
                avg_selected = block_indices_lens.float().mean().item()
                avg_density = avg_selected / num_k_blocks
                print(f"[LongCat] density={avg_density:.4f} avg_selected={avg_selected:.1f}/{num_k_blocks} "
                      f"L={L_orig} chunk=({tq},{hq},{wq}) blocks=({num_q_blocks},{num_k_blocks})")

                # --- Optional: dump real Q/K/V + mask for offline NCU profiling ---
                if os.environ.get("LONGCAT_DUMP", "0") == "1" and FlashAttentionImpl.layeridx == 0 and FlashAttentionImpl.count == 0:
                    dump_dir = os.environ.get("LONGCAT_DUMP_DIR", "/tmp/longcat_dump")
                    os.makedirs(dump_dir, exist_ok=True)
                    _md = torch.zeros(H_lc, num_q_blocks, num_k_blocks, dtype=torch.float32, device='cuda')
                    _cr = torch.arange(block_indices.shape[-1], device='cuda').view(1, 1, 1, -1)
                    _valid = _cr < block_indices_lens[0:1].unsqueeze(-1)
                    _vi = block_indices[0:1].clamp(0, num_k_blocks - 1)
                    _md.unsqueeze(0).scatter_(-1, _vi.long(), _valid.float())
                    torch.save({
                        'q_3d': q_3d.cpu(), 'k_3d': k_3d.cpu(), 'v_3d': v_3d.cpu(),
                        'block_indices': block_indices.cpu(), 'block_indices_lens': block_indices_lens.cpu(),
                        'mask_dense': _md.cpu(), 'latent_shape': lshape, 'chunk_3d': (tq, hq, wq),
                        'num_q_blocks': num_q_blocks, 'num_k_blocks': num_k_blocks,
                        'B': batch_size_lc, 'H': H_lc, 'D': D_lc,
                    }, os.path.join(dump_dir, "longcat_layer0.pt"))
                    print(f"[LongCat] DUMPED real QKV + mask to {dump_dir}/longcat_layer0.pt")

                # Accumulate FLOPs for MFU
                dense_flops_this_call = 4.0 * H_lc * L_orig * L_orig * D_lc
                sparse_flops_this_call = dense_flops_this_call * avg_density
                FlashAttentionImpl.total_dense_flops += dense_flops_this_call
                FlashAttentionImpl.total_attn_flops += sparse_flops_this_call
                FlashAttentionImpl.sum_density += avg_density
                FlashAttentionImpl.density_count += 1

                # --- 7. Convert to FA4 blocksparse format ---
                BLK_lc = chunk_size_q  # block size for FA4 = 128
                cnt_4d, idx_4d = _longcat_indices_to_fa4_blocksparse(
                    block_indices, block_indices_lens, num_k_blocks, batch_size_lc
                )

                # --- 8. Transpose back to FA4 expected format (B, L, H, D) ---
                q_fa4 = q_3d.transpose(1, 2).contiguous()   # (B, L_3d, H, D)
                k_fa4 = k_3d.transpose(1, 2).contiguous()
                v_fa4 = v_3d.transpose(1, 2).contiguous()

                bst = BlockSparseTensorsTorch(
                    full_block_cnt=cnt_4d,
                    full_block_idx=idx_4d,
                    mask_block_cnt=torch.zeros_like(cnt_4d),
                    mask_block_idx=torch.zeros(batch_size_lc, H_lc, num_q_blocks, num_k_blocks,
                                               dtype=torch.int32, device='cuda'),
                    block_size=(BLK_lc, BLK_lc),
                )

                s3_lc = torch.cuda.Event(enable_timing=True)
                s4_lc = torch.cuda.Event(enable_timing=True)
                s3_lc.record()

                out_3d, _ = _fa4_fwd(
                    q_fa4, k_fa4, v_fa4,
                    block_sparse_tensors=bst,
                )

                s4_lc.record()
                torch.cuda.synchronize()
                FlashAttentionImpl.total_attn_ms += s3_lc.elapsed_time(s4_lc)

                if isinstance(out_3d, tuple):
                    out_3d = out_3d[0]

                # --- 9. Rearrange output back: (B, L_3d, H, D) → (B, H, L_3d, D) → THW → (B, L, H, D) ---
                out_bhld = out_3d.transpose(1, 2).contiguous()  # (B, H, L_3d, D)
                out_thw = _longcat_rearrange_3d_block_to_THW(out_bhld, Ntq, Nhq, Nwq, tq, hq, wq)
                out_original = out_thw.transpose(1, 2).contiguous()  # (B, L, H, D)

                FlashAttentionImpl.count += 1
                FlashAttentionImpl.layeridx += 1
                if FlashAttentionImpl.layeridx >= FlashAttentionImpl.num_layers:
                    FlashAttentionImpl.layeridx = 0
                return out_original
            elif sparse_mode == "bsa_cdf":
                # BSA_ATTN CDF: pruned Q (50%) → softmax → cumsum → threshold + min_kv_blocks=4
                import torch.nn.functional as F_bsa
                q_blocked = q_est.view(batch_size, H, MB, BLK, D)
                k_blocked = k_est.view(batch_size, H, NB, BLK, D)
                # Q pruning: keep 50% least similar tokens to block center
                center = q_blocked[:, :, :, BLK // 2:BLK // 2 + 1, :]
                q_norm_bsa = F_bsa.normalize(q_blocked.float(), dim=-1)
                c_norm_bsa = F_bsa.normalize(center.float(), dim=-1)
                sim_bsa = (q_norm_bsa * c_norm_bsa).sum(dim=-1)  # (B, H, MB, BLK)
                keep_size = max(1, BLK // 2)
                _, prune_idx = sim_bsa.topk(keep_size, dim=3, largest=False)
                prune_idx_exp = prune_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
                pruned_q = torch.gather(q_blocked.float(), 3, prune_idx_exp)
                q_c = pruned_q.mean(dim=3).to(q_est.dtype)  # (B, H, MB, D)
                k_c = k_blocked.float().mean(dim=3).to(k_est.dtype)
                scores = torch.matmul(q_c, k_c.transpose(-1, -2)) / (D ** 0.5)
                block_attn = torch.softmax(scores[0], dim=-1)  # (H, MB, NB)
                sorted_attn, sorted_idx = block_attn.sort(dim=-1, descending=True)
                cumsum = sorted_attn.cumsum(dim=-1)
                keep_sorted = torch.ones_like(cumsum, dtype=torch.bool)
                keep_sorted[..., 1:] = cumsum[..., :-1] < threshold
                min_kv_blocks = 4
                min_mask = torch.zeros_like(keep_sorted)
                min_mask[..., :min(min_kv_blocks, NB)] = True
                keep_sorted = keep_sorted | min_mask
                mask_all_heads = torch.zeros((H, MB, NB), dtype=torch.int32, device='cuda')
                mask_bool = torch.zeros((H, MB, NB), dtype=torch.bool, device='cuda')
                mask_bool.scatter_(-1, sorted_idx, keep_sorted)
                mask_all_heads = mask_bool.int()
            elif sparse_mode == "vmoba_thr":
                # VMOBA threshold (query_head mode): normalized gate → cumsum → threshold
                q_blocked = q_est.view(batch_size, H, MB, BLK, D)
                k_blocked = k_est.view(batch_size, H, NB, BLK, D)
                q_c = q_blocked.float().mean(dim=3).to(q_est.dtype)
                k_c = k_blocked.float().mean(dim=3).to(k_est.dtype)
                gate = torch.matmul(q_c, k_c.transpose(-1, -2))[0] / (D ** 0.5)  # (H, MB, NB)
                eps = 1e-6
                row_min = gate.amin(dim=-1, keepdim=True)
                row_max = gate.amax(dim=-1, keepdim=True)
                denom = (row_max - row_min).clamp(min=eps)
                gate_norm = (gate - row_min) / denom
                mn = min(MB, NB)
                diag_arange = torch.arange(mn, device='cuda')
                diag_vals = gate_norm[:, diag_arange, diag_arange]  # (H, mn)
                total_norm = gate_norm.sum(dim=-1)  # (H, MB)
                self_ratio = diag_vals / (total_norm[:, :mn] + eps)
                remain_ratio = (threshold - self_ratio).clamp(min=0)  # (H, mn)
                others_norm = gate_norm.clone()
                others_norm[:, diag_arange, diag_arange] = 0.0
                sorted_norm, sorted_idx = others_norm.sort(dim=-1, descending=True)
                cumsum_v = sorted_norm.cumsum(dim=-1)
                ratio = cumsum_v / (total_norm.unsqueeze(-1) + eps)
                cond = ratio >= remain_ratio.unsqueeze(-1)
                has_cutoff = cond.any(dim=-1)
                cutoff = torch.where(has_cutoff, cond.float().argmax(dim=-1),
                                     torch.full_like(has_cutoff, NB - 1, dtype=torch.long))
                idx_range = torch.arange(NB, device='cuda').view(1, 1, NB)
                sorted_mask = idx_range <= cutoff.unsqueeze(-1)
                mask_bool = torch.zeros((H, MB, NB), dtype=torch.bool, device='cuda')
                mask_bool.scatter_(-1, sorted_idx, sorted_mask)
                mask_bool[:, diag_arange, diag_arange] = True
                mask_all_heads = mask_bool.int()
            else:
                assert sparse_mode == "sparge"
                fm = get_block_map_meansim(
                    q_est, k_est, is_causal=False, BLKQ=BLK, BLKK=BLK,
                    simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=None,
                    return_lut=False, attention_sink=True)
                mask_all_heads = fm[0].to(torch.int32)

            # Enforce diagonal attention per head (local context is always important)
            min_dim = min(MB, NB)
            diag_idx = torch.arange(min_dim, device='cuda')
            mask_all_heads[:, diag_idx, diag_idx] = 1

            avg_density = mask_all_heads.float().mean().item()
            total_blocks = MB * NB
            print(f"[Sparse] per-head avg density: {avg_density:.4f} (avg {mask_all_heads.sum(dim=(1,2)).float().mean().item():.0f}/{total_blocks} blocks)")
            # Accumulate FLOPs for MFU: attention FLOPs = 4 * H * L^2 * D (QK + PV)
            dense_flops_this_call = 4.0 * H * L_padded * L_padded * D
            sparse_flops_this_call = dense_flops_this_call * avg_density
            FlashAttentionImpl.total_dense_flops += dense_flops_this_call
            FlashAttentionImpl.total_attn_flops += sparse_flops_this_call
            FlashAttentionImpl.sum_density += avg_density
            FlashAttentionImpl.density_count += 1
            # 每个head内部的 nnz 分布统计（按行统计）:
            # row_nnz[h, i] = 第h个head第i行的非零block个数
            row_nnz = mask_all_heads.sum(dim=2).float()  # (H, MB)

            # 每个head一个 std / mean / cv
            per_head_std = row_nnz.std(dim=1, unbiased=False)                     # (H,)
            per_head_mean = row_nnz.mean(dim=1)                                   # (H,)
            per_head_cv = per_head_std / per_head_mean.clamp(min=1e-8)            # (H,)

            # 单行打印：h0(std=...,cv=...) | h1(...) | ...
            std_cv_line = " | ".join(
                [f"h{i}(std={s:.4f},cv={c:.4f})"
                for i, (s, c) in enumerate(zip(per_head_std.detach().cpu().tolist(),
                                                per_head_cv.detach().cpu().tolist()))]
            )
            print(f"[Sparse] per-head row-nnz stats -> {std_cv_line}", flush=True)
            
            s2.record()
            # 4. Per-head reordering — VECTORIZED
            s_r1 = torch.cuda.Event(enable_timing=True)
            s_r2 = torch.cuda.Event(enable_timing=True)
            s_r3 = torch.cuda.Event(enable_timing=True)
            s_r4 = torch.cuda.Event(enable_timing=True)
            s_r5 = torch.cuda.Event(enable_timing=True)
            s_r1.record()

            if mode == "dump_mask":
                global MASK_DUMP_IDX
                mask_dir = os.environ.get("VLLM_MASK_DIR", "/data1/dhelix_shared/daizijian/vllm-omni-mask")
                os.makedirs(mask_dir, exist_ok=True)
                thr_str = str(threshold).replace('.', '_')
                sub_dir = os.path.join(mask_dir, f"mask_{MASK_DUMP_IDX}_thr{thr_str}")
                os.makedirs(sub_dir, exist_ok=True)
                # 保存原始mask
                orig_path = os.path.join(sub_dir, f"mask_no_reorder.pt")
                try:
                    torch.save(mask_all_heads.detach().cpu(), orig_path)
                except Exception:
                    pass
                # 遍历所有重排方法
                reorder_methods = [
                    ("with_reorder", lambda m: batched_chunked_greedy_multihead(m.float(), chunk_size=32)),
                    ("bcg_reorder", lambda m: global_banded_greedy_reorder_cuda(m.float())),
                    ("opt_reorder", lambda m: optimal_reorder_cuda(m.float())),
                    ("fast_greedy", lambda m: global_greedy_reorder_cuda(m.float())),
                    ("numba_reorder", lambda m: torch.stack([reorder_rows_by_similarity(m[h], metric="intersection") for h in range(m.shape[0])])),
                    ("nnz_reorder", lambda m: torch.argsort(m.float().sum(dim=2), descending=True, dim=1)),
                    ("wave_reorder", lambda m: wave_overlap_reorder(m.float(), wave_size=132)),
                    ("wave_1d_reorder", lambda m: __import__("vllm_omni.diffusion.attention.backends.greedy_reorder_cuda", fromlist=["wave_1d_reorder"]).wave_1d_reorder(m.float(), wave_size=132)),
                    ("wave_2d_reorder", lambda m: __import__("vllm_omni.diffusion.attention.backends.greedy_reorder_cuda", fromlist=["wave_2d_reorder"]).wave_2d_reorder(m.float(), wave_size=132)),
                    ("hybrid_reorder", lambda m: __import__("vllm_omni.diffusion.attention.backends.greedy_reorder_cuda", fromlist=["hybrid_block_reorder"]).hybrid_block_reorder(m.float(), wave_size=132, waves_per_block=4)),
                    ("nnz_3step", lambda m: nnz_3step_reorder(m.float(), wave_size=132)),
                ]
                for method_name, method_func in reorder_methods:
                    try:
                        perm_all_heads = method_func(mask_all_heads)
                        perm_for_mask = perm_all_heads.unsqueeze(-1).expand_as(mask_all_heads)
                        reordered_masks = torch.gather(mask_all_heads, 1, perm_for_mask).bool()
                        perm_path = os.path.join(sub_dir, f"mask_{method_name}_perm.pt")
                        torch.save(reordered_masks.detach().cpu(), perm_path)
                    except Exception as e:
                        print(f"[DUMP_MASK] {method_name} failed: {e}")
                MASK_DUMP_IDX += 1
                perm_all_heads = torch.arange(MB, device='cuda').unsqueeze(0).expand(H, MB).contiguous()
            elif mode == "with_reorder":
                # Batch all H heads' greedy sort into one fused operation
                perm_all_heads = batched_chunked_greedy_multihead(mask_all_heads.float(), chunk_size=32)
            elif mode == "bcg_l2_reorder":
                perm_all_heads = bcg_l2_reorder_cuda(mask_all_heads.float(), window=256)
            elif mode == "bcg_reorder":
                # CUDA global greedy: single kernel launch, all heads parallel, ~82ms
                perm_all_heads = global_banded_greedy_reorder_cuda(mask_all_heads.float())
            elif mode == "opt_reorder":
                perm_all_heads = optimal_reorder_cuda(mask_all_heads.float())
            elif mode == "fast_greedy":
                # CUDA global greedy: single kernel launch, all heads parallel, ~82ms
                perm_all_heads = global_greedy_reorder_cuda(mask_all_heads.float())
            elif mode == "numba_reorder":
                perm_all_heads = torch.zeros((H, MB), dtype=torch.long, device='cuda')
                for h in range(H):
                    perm_all_heads[h] = reorder_rows_by_similarity(mask_all_heads[h], metric="intersection")
            elif mode == "nnz_reorder":
                # Fully vectorized: argsort across all heads at once
                perm_all_heads = torch.argsort(mask_all_heads.float().sum(dim=2), descending=True, dim=1)
            elif mode == "wave_reorder":
                # Wave-overlap clustering: 132-row waves with max column overlap
                # NCU-verified: -55% DRAM read vs nnz_reorder
                perm_all_heads = wave_overlap_reorder(mask_all_heads.float(), wave_size=132)
            elif mode == "wave_1d_reorder":
                from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import wave_1d_reorder
                perm_all_heads = wave_1d_reorder(mask_all_heads.float(), wave_size=132)
            elif mode == "nnz_3step" or mode == "nnz_3step_reorder":
                perm_all_heads = nnz_3step_reorder(mask_all_heads.float(), wave_size=132)
            elif mode == "svd_deflation_reorder":
                from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import svd_deflation_reorder
                perm_all_heads = svd_deflation_reorder(mask_all_heads.float(), wave_size=132)
            elif mode in ("svd_cumulative_rib_reorder", "nnz_head_aligned_reorder"):
                if (batch_size != 1 or self.causal
                        or (BLK, D) != (128, 128) or key.shape[2] != H
                        or torch.cuda.get_device_capability(query.device) != (9, 0)
                        or os.environ.get("VLLM_KV_ORDER", "asc") != "asc"):
                    raise ValueError(
                        "head-aligned reorder requires noncausal SM90 FA4, "
                        "B128/D128, batch1, equal heads and asc KV indices"
                    )
                from vllm_omni.diffusion.attention.backends.cumulative_svd_rib import reorder
                perm_all_heads = reorder(
                    mask_all_heads.bool().contiguous(), stage2_only=mode == "nnz_head_aligned_reorder"
                )
            elif mode == "svd_u2u3_reorder":
                from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import svd_u2u3_reorder
                perm_all_heads = svd_u2u3_reorder(mask_all_heads.float(), wave_size=132)
            elif mode == "wave_2d_reorder":
                from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import wave_2d_reorder
                perm_all_heads = wave_2d_reorder(mask_all_heads.float(), wave_size=132)
            elif mode == "hybrid_reorder":
                from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import hybrid_block_reorder
                perm_all_heads = hybrid_block_reorder(mask_all_heads.float(), wave_size=132, waves_per_block=4)
            else:
                perm_all_heads = torch.arange(MB, device='cuda').unsqueeze(0).expand(H, MB).contiguous()
            
            s_r2.record()
            # Per-head Q reorder — VECTORIZED via gather
            # q_blocked: (batch, MB, BLK, H, D)
            # perm_all_heads: (H, MB) → need gather index (batch, MB, BLK, H, D) along dim=1
            q_blocked = query.view(batch_size, MB, BLK, H, D)
            # perm_all_heads[h, j] → index[b, j, blk, h, d] = perm_all_heads[h, j]
            gather_idx = perm_all_heads.T.unsqueeze(0).unsqueeze(2).unsqueeze(4).expand(batch_size, MB, BLK, H, D)
            q_reordered = torch.gather(q_blocked, 1, gather_idx).view(batch_size, L_padded, H, D).contiguous()
            
            s_r3.record()
            # Build per-head block-sparse dict — VECTORIZED
            # Reorder masks: reordered_masks[h] = mask_all_heads[h][perm_all_heads[h]]
            perm_for_mask = perm_all_heads.unsqueeze(-1).expand(H, MB, NB)
            reordered_masks = torch.gather(mask_all_heads, 1, perm_for_mask).bool()  # (H, MB, NB)
            
            # 其它模式下不再单独保存mask
            # Count NNZ per row per head
            cnt_all = reordered_masks.sum(dim=-1, dtype=torch.int32)  # (H, MB)
            
            # Sort columns: True cols first (argsort on ~mask puts True=0 first)
            sorted_cols_all = torch.argsort(~reordered_masks, dim=2, stable=True).int()  # (H, MB, NB)
            
            # Zero out invalid entries beyond cnt (FA4 likely ignores them, but be safe)
            col_range = torch.arange(NB, device='cuda').view(1, 1, NB)  # (1, 1, NB)
            valid = col_range < cnt_all.unsqueeze(-1)  # (H, MB, NB)
            sorted_cols_all = sorted_cols_all * valid.int()
            
            s_r4.record()
            # Expand to (batch, H, MB, ...) format
            cnt_4d = cnt_all.unsqueeze(0).expand(batch_size, H, MB).contiguous()
            idx_4d = sorted_cols_all.unsqueeze(0).expand(batch_size, H, MB, NB).contiguous()
            
            sp = {
                "full_block_cnt": cnt_4d,
                "full_block_idx": idx_4d,
                "mask_block_cnt": torch.zeros_like(cnt_4d),
                "mask_block_idx": torch.zeros(batch_size, H, MB, NB, dtype=torch.int32, device='cuda'),
                "block_size": (BLK, BLK),
            }
            s_r5.record()
            
            s3.record()
            if BLK != 128:
                # Use low-level FA4 API for non-128 block sizes
                bst = BlockSparseTensorsTorch(
                    full_block_cnt=sp["full_block_cnt"],
                    full_block_idx=sp["full_block_idx"],
                    mask_block_cnt=sp["mask_block_cnt"],
                    mask_block_idx=sp["mask_block_idx"],
                    block_size=(BLK, BLK),
                )
                out_reordered, _ = _fa4_fwd(
                    q_reordered, key, value,
                    tile_mn=(BLK, BLK),
                    block_sparse_tensors=bst,
                )
            else:
                out_reordered = flash_attn_func_for_sparse_reorder(
                    q_reordered, key, value,
                    **sp
                )
            s4.record()
            
            if isinstance(out_reordered, tuple):
                out_reordered = out_reordered[0]

            # FA4 output is (batch, seqlen, H, D), revert per-head reorder — VECTORIZED via scatter
            out_blocked = out_reordered.view(batch_size, MB, BLK, H, D)
            out_original = torch.zeros_like(out_blocked)
            # gather_idx was (batch, MB, BLK, H, D) with perm indices along dim=1
            out_original.scatter_(1, gather_idx, out_blocked)
            out_original = out_original.view(batch_size, L_padded, H, D)
            
            
            # Remove Padding
            if pad_len > 0:
                out_original = out_original[:, :-pad_len, :, :]
            s5.record()
            torch.cuda.synchronize()
            FlashAttentionImpl.total_sparse_ms += s1.elapsed_time(s2)
            FlashAttentionImpl.total_reorder_ms += s2.elapsed_time(s3)
            FlashAttentionImpl.total_attn_ms += s3.elapsed_time(s4)
            FlashAttentionImpl.total_o_shuffle_ms += s4.elapsed_time(s5)
            FlashAttentionImpl.total_r_algo_ms += s_r1.elapsed_time(s_r2)
            FlashAttentionImpl.total_r_qgather_ms += s_r2.elapsed_time(s_r3)
            FlashAttentionImpl.total_r_sparse_ms += s_r3.elapsed_time(s_r4)
            FlashAttentionImpl.total_r_expand_ms += s_r4.elapsed_time(s_r5)
            FlashAttentionImpl.count += 1
            return out_original



        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        return self._forward_varlen_dense(
            query,
            key,
            value,
        )

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """XPU flash attention implementation."""
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            HAS_FLASH_ATTN,
        )

        if not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionBackend requires Flash Attention. "
                "Please assure vllm-xpu-kernels properly installed. "
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        return self._forward_varlen_dense(
            query,
            key,
            value,
        )

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """NPU attention implementation using mindiesd."""
        try:
            from mindiesd import attention_forward
        except ImportError:
            raise ImportError(
                "FlashAttentionBackend NPU implementation requires MindIE-SD. "
                "Please install MindIE-SD to enable NPU attention support. "
                "For installation details, see https://gitcode.com/Ascend/MindIE-SD"
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata else None
        output = attention_forward(
            query,
            key,
            value,
            attn_mask=attention_mask,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout="BNSD",
        )
        return output

def print_final_stats():
    # 注意：这里要通过类名访问
    total = FlashAttentionImpl.total_reorder_ms + FlashAttentionImpl.total_attn_ms
    if total > 0:
        print("\n" + "!"*20 + " 最终性能报告 " + "!"*20)
        # 修改你的 print 语句
        print(f"[MY_LOG] 总计稀疏 (Sparse): {FlashAttentionImpl.total_sparse_ms:.2f} ms")
        print(f"[MY_LOG] 总计重排 (Reorder): {FlashAttentionImpl.total_reorder_ms:.2f} ms")
        print(f"[MY_LOG]   ├─ algo:    {FlashAttentionImpl.total_r_algo_ms:.2f} ms")
        print(f"[MY_LOG]   ├─ Q_gather: {FlashAttentionImpl.total_r_qgather_ms:.2f} ms")
        print(f"[MY_LOG]   ├─ sparse:  {FlashAttentionImpl.total_r_sparse_ms:.2f} ms")
        print(f"[MY_LOG]   └─ expand:  {FlashAttentionImpl.total_r_expand_ms:.2f} ms")
        print(f"[MY_LOG] 总计算子 (Pure Attn): {FlashAttentionImpl.total_attn_ms:.2f} ms")
        print(f"[MY_LOG] 总计O_shuffle (O_shuffle): {FlashAttentionImpl.total_o_shuffle_ms:.2f} ms")
        print(f"[MY_LOG] 总计count {FlashAttentionImpl.count}")

        # ── MFU Report ──
        attn_ms = FlashAttentionImpl.total_attn_ms
        peak_flops_per_sec = FlashAttentionImpl.H100_BF16_PEAK_TFLOPS * 1e12  # FLOPS
        if attn_ms > 0 and FlashAttentionImpl.density_count > 0:
            avg_density = FlashAttentionImpl.sum_density / FlashAttentionImpl.density_count
            attn_sec = attn_ms / 1000.0
            # Dense-equivalent MFU: pretend all FLOPs were dense
            dense_mfu = FlashAttentionImpl.total_dense_flops / (attn_sec * peak_flops_per_sec)
            # Actual sparse MFU: only count the FLOPs actually computed
            sparse_mfu = FlashAttentionImpl.total_attn_flops / (attn_sec * peak_flops_per_sec)
            # Per-call averages
            avg_attn_ms = attn_ms / FlashAttentionImpl.count if FlashAttentionImpl.count > 0 else 0
            avg_dense_tflops = (FlashAttentionImpl.total_dense_flops / FlashAttentionImpl.count / 1e12) if FlashAttentionImpl.count > 0 else 0
            avg_sparse_tflops = (FlashAttentionImpl.total_attn_flops / FlashAttentionImpl.count / 1e12) if FlashAttentionImpl.count > 0 else 0

            print(f"\n{'─'*20} Attention Kernel MFU {'─'*20}")
            print(f"[MFU] Avg block density:     {avg_density:.4f}")
            print(f"[MFU] Total dense FLOPs:     {FlashAttentionImpl.total_dense_flops:.3e}  (若全量计算)")
            print(f"[MFU] Total sparse FLOPs:    {FlashAttentionImpl.total_attn_flops:.3e}  (实际计算)")
            print(f"[MFU] Attn kernel time:      {attn_ms:.2f} ms ({attn_sec:.4f} s)")
            print(f"[MFU] Avg per-call latency:  {avg_attn_ms:.2f} ms")
            print(f"[MFU] ──────── MFU = sparse_FLOPs / (peak × time) ────────")
            print(f"[MFU] MFU:                   {sparse_mfu*100:.2f}%  ({FlashAttentionImpl.total_attn_flops / attn_sec / 1e12:.2f} / {FlashAttentionImpl.H100_BF16_PEAK_TFLOPS:.1f} TFLOPS)")
            print(f"[MFU] Achieved TFLOPS/s:     {FlashAttentionImpl.total_attn_flops / attn_sec / 1e12:.2f}  (实际稀疏吞吐)")
            print(f"[MFU] ──────── Effective Throughput (dense-equiv) ────────")
            print(f"[MFU] Eff. TFLOPS/s:         {FlashAttentionImpl.total_dense_flops / attn_sec / 1e12:.2f}  (dense_FLOPs / attn_time)")
            print(f"[MFU] Eff. Speedup vs dense: {dense_mfu:.2f}×  (等效加速比: 用稀疏时间完成了多少dense工作量)")
            print(f"[MFU] H200 BF16 peak:        {FlashAttentionImpl.H100_BF16_PEAK_TFLOPS:.1f} TFLOPS")
            print(f"{'─'*61}")
        print("!"*54 + "\n")

# 注册钩子：当 Python 进程退出时自动执行该函数
atexit.register(print_final_stats)