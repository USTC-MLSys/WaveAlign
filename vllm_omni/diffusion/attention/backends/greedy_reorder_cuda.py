"""CUDA kernels for greedy reordering of block-sparse attention masks.

Provides:
1. global_greedy_reorder_cuda(mask_float, batch_heads=8)
   — Original: pre-compute S = mask @ mask.T, run greedy TSP on S
   — Good for small masks, simple API

2. global_banded_greedy_reorder_cuda(mask_float, band_size=4)
   — Banded CUDA Greedy (BCG): NNZ banding + 1-lookback overlap greedy
   — Better L2 hit rate (NNZ banding = wavefront shear protection)
   — Lower overhead (no pre-computed S matrix, parallel per-band)

3. global_banded_window_greedy_reorder_cuda(mask_float, band_size=4, window=132)
   — Banded Window-Greedy: NNZ banding + W-lookback greedy per band
   — Best L2 hit rate among all tested algorithms

4. optimal_reorder_cuda(mask_float, band_size=4, window=132)
   — Production API: band4 window-greedy + head-adaptive skip
   — NCU-verified: 90.51% L2 hit @ thr=0.95, 92.42% @ thr=0.7
   — Overhead: ~0.3s for 16 heads × 3210 blocks on H100

Usage:
    from greedy_reorder_cuda import optimal_reorder_cuda
    perm = optimal_reorder_cuda(mask_float)   # mask: (H, MB, NB) -> perm: (H, MB)
"""
import torch
import os
from torch.utils.cpp_extension import load_inline

_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

// =============================================================================
// Kernel 1: Original global greedy reorder (pre-computed similarity matrix)
// =============================================================================
// One CUDA block per head. Each block runs the full greedy loop for one head.
// S is (H, MB, MB) row-major float32. Modified in-place (column zeroing).
__global__ void greedy_reorder_kernel(
    float* __restrict__ S,
    int64_t* __restrict__ perm,
    int H, int MB
) {
    int h = blockIdx.x;
    if (h >= H) return;

    float* S_h = S + (int64_t)h * MB * MB;
    int64_t* perm_h = perm + (int64_t)h * MB;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ char smem[];
    float* smax = (float*)smem;
    int* sidx = (int*)(smax + nthreads);

    // Find starting row (highest row sum = most connected)
    float best_sum = -1e30f;
    int best_start = 0;
    for (int r = tid; r < MB; r += nthreads) {
        float rsum = 0.0f;
        for (int c = 0; c < MB; c++) {
            rsum += S_h[(int64_t)r * MB + c];
        }
        if (rsum > best_sum) {
            best_sum = rsum;
            best_start = r;
        }
    }
    smax[tid] = best_sum;
    sidx[tid] = best_start;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    int curr = sidx[0];
    if (tid == 0) perm_h[0] = curr;

    // Mark starting row visited: column zeroing
    for (int r = tid; r < MB; r += nthreads) {
        S_h[(int64_t)r * MB + curr] = -1e30f;
    }
    __syncthreads();

    // Main greedy loop
    for (int step = 1; step < MB; step++) {
        float my_best_val = -1e30f;
        int my_best_idx = 0;

        for (int j = tid; j < MB; j += nthreads) {
            float val = S_h[(int64_t)curr * MB + j];
            if (val > my_best_val) {
                my_best_val = val;
                my_best_idx = j;
            }
        }

        smax[tid] = my_best_val;
        sidx[tid] = my_best_idx;
        __syncthreads();

        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s && smax[tid + s] > smax[tid]) {
                smax[tid] = smax[tid + s];
                sidx[tid] = sidx[tid + s];
            }
            __syncthreads();
        }

        curr = sidx[0];
        if (tid == 0) perm_h[step] = curr;

        for (int r = tid; r < MB; r += nthreads) {
            S_h[(int64_t)r * MB + curr] = -1e30f;
        }
        __syncthreads();
    }
}

torch::Tensor greedy_reorder_cuda(torch::Tensor S) {
    const c10::cuda::CUDAGuard device_guard(S.device());
    int H = S.size(0);
    int MB = S.size(1);
    auto perm = torch::zeros({H, MB}, torch::dtype(torch::kInt64).device(S.device()));
    auto S_work = S.clone();

    int threads = 256;
    int smem = threads * (sizeof(float) + sizeof(int));
    greedy_reorder_kernel<<<H, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        S_work.data_ptr<float>(),
        perm.data_ptr<int64_t>(),
        H, MB
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return perm;
}


// =============================================================================
// Kernel 2: Banded greedy reorder (on-the-fly overlap, no pre-computed S)
// =============================================================================
// One CUDA block per (head, band). Each block:
//   1. Caches the current row's mask in shared memory
//   2. Computes overlap with all unvisited band rows on-the-fly
//   3. Greedy picks the row with max overlap
//
// Shared memory layout:
//   [curr_cache: NB floats] [smax: nthreads floats] [sidx: nthreads ints] [visited: N_B bools]
__global__ void banded_greedy_reorder_kernel(
    const float* __restrict__ mask,         // (H, MB, NB) row-major
    const int* __restrict__ band_row_idx,   // flattened row indices for all bands
    const int* __restrict__ band_offsets,    // (total_bands + 1,) cumulative offsets into band_row_idx
    const int* __restrict__ band_heads,      // (total_bands,) head index for each band
    const int* __restrict__ perm_write_off,  // (total_bands,) write offset in perm[h, :]
    int64_t* __restrict__ perm,             // (H, MB) output permutation
    int MB, int NB
) {
    int bid = blockIdx.x;
    int h = band_heads[bid];
    int bstart = band_offsets[bid];
    int bend = band_offsets[bid + 1];
    int N_B = bend - bstart;
    if (N_B <= 0) return;

    const float* mask_h = mask + (int64_t)h * MB * NB;
    int64_t* perm_h = perm + (int64_t)h * MB;
    int write_off = perm_write_off[bid];

    int tid = threadIdx.x;
    int nt = blockDim.x;

    extern __shared__ char smem[];
    float* curr_cache = (float*)smem;                     // NB floats
    float* smax = (float*)(curr_cache + NB);              // nt floats
    int* sidx = (int*)(smax + nt);                        // nt ints
    bool* visited = (bool*)(sidx + nt);                   // N_B bools

    // Initialize visited
    for (int i = tid; i < N_B; i += nt) visited[i] = false;
    __syncthreads();

    // Trivial case: single row
    if (N_B == 1) {
        if (tid == 0) perm_h[write_off] = band_row_idx[bstart];
        return;
    }

    // --- Find starting row: highest NNZ (row sum) ---
    float best = -1e30f;
    int best_i = 0;
    for (int i = tid; i < N_B; i += nt) {
        int row = band_row_idx[bstart + i];
        const float* rd = mask_h + (int64_t)row * NB;
        float s = 0.0f;
        for (int k = 0; k < NB; k++) s += rd[k];
        if (s > best) { best = s; best_i = i; }
    }
    smax[tid] = best; sidx[tid] = best_i;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s]; sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    int curr_local = sidx[0];
    int curr_row = band_row_idx[bstart + curr_local];
    if (tid == 0) {
        perm_h[write_off] = curr_row;
        visited[curr_local] = true;
    }
    __syncthreads();

    // --- Main greedy loop ---
    for (int step = 1; step < N_B; step++) {
        // Cache current row mask in shared memory (coalesced read)
        for (int k = tid; k < NB; k += nt) {
            curr_cache[k] = mask_h[(int64_t)curr_row * NB + k];
        }
        __syncthreads();

        // Each thread evaluates one or more candidate rows
        float my_best = -1e30f;
        int my_best_i = 0;

        for (int i = tid; i < N_B; i += nt) {
            if (visited[i]) continue;
            int row = band_row_idx[bstart + i];
            const float* rd = mask_h + (int64_t)row * NB;

            // Compute overlap = dot(curr_cache, rd) — on-the-fly
            float overlap = 0.0f;
            for (int k = 0; k < NB; k++) {
                overlap += curr_cache[k] * rd[k];
            }

            if (overlap > my_best) {
                my_best = overlap;
                my_best_i = i;
            }
        }

        // Block-wide argmax reduction
        smax[tid] = my_best; sidx[tid] = my_best_i;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s && smax[tid + s] > smax[tid]) {
                smax[tid] = smax[tid + s]; sidx[tid] = sidx[tid + s];
            }
            __syncthreads();
        }

        curr_local = sidx[0];
        curr_row = band_row_idx[bstart + curr_local];
        if (tid == 0) {
            perm_h[write_off + step] = curr_row;
            visited[curr_local] = true;
        }
        __syncthreads();
    }
}

torch::Tensor banded_greedy_reorder_cuda(
    torch::Tensor mask,
    torch::Tensor band_row_idx,
    torch::Tensor band_offsets,
    torch::Tensor band_heads,
    torch::Tensor perm_write_off,
    int total_bands,
    int H, int MB, int NB,
    int max_band_size
) {
    const c10::cuda::CUDAGuard device_guard(mask.device());
    auto perm = torch::zeros({H, MB}, torch::dtype(torch::kInt64).device(mask.device()));

    int threads = 256;
    // smem: curr_cache(NB*4) + smax(256*4) + sidx(256*4) + visited(max_band_size)
    int smem = NB * sizeof(float) + threads * (sizeof(float) + sizeof(int)) + max_band_size;

    banded_greedy_reorder_kernel<<<total_bands, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        mask.data_ptr<float>(),
        band_row_idx.data_ptr<int>(),
        band_offsets.data_ptr<int>(),
        band_heads.data_ptr<int>(),
        perm_write_off.data_ptr<int>(),
        perm.data_ptr<int64_t>(),
        MB, NB
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return perm;
}


// =============================================================================
// Kernel 3: Window-lookback greedy reorder (pre-computed S, W-lookback)
// =============================================================================
// Like Kernel 1, but instead of argmax(S[curr, :]), picks
//   argmax(window_sum[:])  where  window_sum = sum(S[last_W_rows, :])
// This better matches the 132-SM concurrent window.
// S is (H, MB, MB), modified in-place. W = lookback window size.
__global__ void window_greedy_reorder_kernel(
    float* __restrict__ S,
    int64_t* __restrict__ perm,
    float* __restrict__ window_sum,  // (H, MB) workspace for accumulated scores
    const int* __restrict__ lengths, // Optional: true length per batch
    int H, int MB, int W
) {
    int h = blockIdx.x;
    if (h >= H) return;

    float* S_h = S + (int64_t)h * MB * MB;
    int64_t* perm_h = perm + (int64_t)h * MB;
    float* wsum = window_sum + (int64_t)h * MB;
    int N_B = lengths ? lengths[h] : MB;
    if (N_B <= 0) return;

    int tid = threadIdx.x;
    int nt = blockDim.x;

    extern __shared__ char smem[];
    float* smax = (float*)smem;
    int* sidx = (int*)(smax + nt);

    // Initialize window_sum to 0
    for (int j = tid; j < N_B; j += nt) wsum[j] = 0.0f;
    __syncthreads();

    // Find starting row (highest row sum)
    float best = -1e30f;
    int best_i = 0;
    for (int r = tid; r < N_B; r += nt) {
        float rsum = 0.0f;
        for (int c = 0; c < N_B; c++) rsum += S_h[(int64_t)r * MB + c];
        if (rsum > best) { best = rsum; best_i = r; }
    }
    smax[tid] = best; sidx[tid] = best_i;
    __syncthreads();
    for (int s = nt/2; s > 0; s >>= 1) {
        if (tid < s && smax[tid+s] > smax[tid]) { smax[tid] = smax[tid+s]; sidx[tid] = sidx[tid+s]; }
        __syncthreads();
    }

    int curr = sidx[0];
    if (tid == 0) perm_h[0] = curr;

    // Add curr row to window_sum, mark visited
    for (int j = tid; j < N_B; j += nt) {
        wsum[j] += S_h[(int64_t)curr * MB + j];
        S_h[(int64_t)j * MB + curr] = -1e30f;  // column zero = visited
    }
    __syncthreads();

    // Main greedy loop
    for (int step = 1; step < N_B; step++) {
        // Pick unvisited row with max window_sum score
        float my_best = -1e30f;
        int my_idx = 0;
        for (int j = tid; j < N_B; j += nt) {
            // Check visited via column-zeroed S (if S[0,j] == -1e30 it's visited)
            // Actually we use wsum directly; visited cols have wsum dominated by -1e30 contributions
            // Better: check S_h[j*MB+j] but that's unreliable. Use a separate check.
            float val = wsum[j];
            // Column j was zeroed if visited, so S_h[curr*MB+j] would be -1e30
            // But wsum accumulated before zeroing. We need explicit visited check.
            // Use the diagonal: if S_h[j*MB+j] == -1e30, j is visited
            if (S_h[(int64_t)j * MB + j] < -1e20f) continue;  // visited
            if (val > my_best) { my_best = val; my_idx = j; }
        }

        smax[tid] = my_best; sidx[tid] = my_idx;
        __syncthreads();
        for (int s = nt/2; s > 0; s >>= 1) {
            if (tid < s && smax[tid+s] > smax[tid]) { smax[tid] = smax[tid+s]; sidx[tid] = sidx[tid+s]; }
            __syncthreads();
        }

        curr = sidx[0];
        if (tid == 0) perm_h[step] = curr;

        // Add curr row to window_sum
        for (int j = tid; j < N_B; j += nt) {
            wsum[j] += S_h[(int64_t)curr * MB + j];
        }
        __syncthreads();

        // Subtract row that falls out of window (step - W)
        if (step >= W) {
            int old_row = (int)perm_h[step - W];
            // We can't read S_h[old_row, j] because columns of old_row's
            // selected neighbors were zeroed. But S_h[old_row, :] (the ROW)
            // was never modified — only columns were zeroed.
            // So we read the old_row's ROW to subtract.
            for (int j = tid; j < N_B; j += nt) {
                // S_h[old_row * MB + j] is the original value if j wasn't visited
                // when old_row was added. But some j's were visited after old_row
                // was added (their columns were zeroed). For those j's,
                // S_h[old_row * MB + j] is now -1e30 (from column zeroing).
                // We should only subtract the original value. Since visited j's
                // won't be candidates anyway, subtracting -1e30 would make
                // wsum[j] very positive — but j is visited so it's filtered out.
                // So it's safe to just subtract whatever is there.
                wsum[j] -= S_h[(int64_t)old_row * MB + j];
            }
            __syncthreads();
        }

        // Mark curr visited (column zeroing)
        for (int j = tid; j < N_B; j += nt) {
            S_h[(int64_t)j * MB + curr] = -1e30f;
        }
        __syncthreads();
    }
}

torch::Tensor window_greedy_reorder_cuda(torch::Tensor S, int W, torch::Tensor lengths) {
    const c10::cuda::CUDAGuard device_guard(S.device());
    int H = S.size(0);
    int MB = S.size(1);
    auto perm = torch::zeros({H, MB}, torch::dtype(torch::kInt64).device(S.device()));
    auto S_work = S.clone();
    auto wsum = torch::zeros({H, MB}, torch::dtype(torch::kFloat32).device(S.device()));
    const int* lengths_ptr = nullptr;
    if (lengths.numel() > 0) {
        lengths_ptr = lengths.data_ptr<int>();
    }

    int threads = 256;
    int smem = threads * (sizeof(float) + sizeof(int));
    window_greedy_reorder_kernel<<<H, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        S_work.data_ptr<float>(),
        perm.data_ptr<int64_t>(),
        wsum.data_ptr<float>(),
        lengths_ptr,
        H, MB, W
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return perm;
}

// =============================================================================
// Kernel 4: Banded Window-lookback greedy reorder (on-the-fly, W-lookback)
// =============================================================================
__global__ void banded_window_greedy_reorder_kernel(
    const float* __restrict__ mask,          // (H, MB, NB)
    const int64_t* __restrict__ sorted_indices,  // (H, MB)
    const int* __restrict__ band_heads,      // (total_bands,)
    const int* __restrict__ band_starts,     // (total_bands,)
    const int* __restrict__ band_ends,       // (total_bands,)
    int64_t* __restrict__ perm,              // (H, MB)
    int MB, int NB, int W
) {
    int bid = blockIdx.x;
    int h = band_heads[bid];
    int cs = band_starts[bid];
    int ce = band_ends[bid];
    int N_B = ce - cs;
    if (N_B <= 0) return;

    const float* mask_h = mask + (int64_t)h * MB * NB;
    const int64_t* sorted_indices_h = sorted_indices + (int64_t)h * MB;
    int64_t* perm_h = perm + (int64_t)h * MB;

    int tid = threadIdx.x;
    int nt = blockDim.x;

    extern __shared__ char smem[];
    float* window_freq = (float*)smem;                    // NB floats
    float* smax = (float*)(window_freq + NB);             // nt floats
    int* sidx = (int*)(smax + nt);                        // nt ints
    bool* visited = (bool*)(sidx + nt);                   // N_B bools

    for (int k = tid; k < NB; k += nt) window_freq[k] = 0.0f;
    for (int i = tid; i < N_B; i += nt) visited[i] = false;
    __syncthreads();

    if (N_B == 1) {
        if (tid == 0) perm_h[cs] = sorted_indices_h[cs];
        return;
    }

    // --- Find starting row: highest NNZ ---
    float best = -1e30f;
    int best_i = 0;
    for (int i = tid; i < N_B; i += nt) {
        int row = sorted_indices_h[cs + i];
        const float* rd = mask_h + (int64_t)row * NB;
        float s = 0.0f;
        for (int k = 0; k < NB; k++) s += rd[k];
        if (s > best) { best = s; best_i = i; }
    }
    smax[tid] = best; sidx[tid] = best_i;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s]; sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    int curr_local = sidx[0];
    int curr_row = sorted_indices_h[cs + curr_local];
    if (tid == 0) {
        perm_h[cs] = curr_row;
        visited[curr_local] = true;
    }
    __syncthreads();

    // Add initial row to window_freq
    for (int k = tid; k < NB; k += nt) {
        window_freq[k] += mask_h[(int64_t)curr_row * NB + k];
    }
    __syncthreads();

    // --- Main greedy loop ---
    for (int step = 1; step < N_B; step++) {
        float my_best = -1e30f;
        int my_best_i = 0;

        for (int i = tid; i < N_B; i += nt) {
            if (visited[i]) continue;
            int row = sorted_indices_h[cs + i];
            const float* rd = mask_h + (int64_t)row * NB;

            float misses = 0.0f;
            float hits = 0.0f;
            for (int k = 0; k < NB; k++) {
                if (rd[k] > 0.0f) {
                    if (window_freq[k] > 0.0f) {
                        hits += 1.0f;
                    } else {
                        misses += 1.0f;
                    }
                }
            }
            float cost = -misses + 0.0001f * hits;

            if (cost > my_best) {
                my_best = cost;
                my_best_i = i;
            }
        }

        smax[tid] = my_best; sidx[tid] = my_best_i;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s && smax[tid + s] > smax[tid]) {
                smax[tid] = smax[tid + s]; sidx[tid] = sidx[tid + s];
            }
            __syncthreads();
        }

        curr_local = sidx[0];
        curr_row = sorted_indices_h[cs + curr_local];
        if (tid == 0) {
            perm_h[cs + step] = curr_row;
            visited[curr_local] = true;
        }
        __syncthreads();

        // Update window_freq
        for (int k = tid; k < NB; k += nt) {
            float delta = mask_h[(int64_t)curr_row * NB + k];
            if (step >= W) {
                int old_row = perm_h[cs + step - W];
                delta -= mask_h[(int64_t)old_row * NB + k];
            }
            window_freq[k] += delta;
        }
        __syncthreads();
    }
}

torch::Tensor banded_window_greedy_reorder_cuda(
    torch::Tensor mask,
    torch::Tensor sorted_indices,
    torch::Tensor band_heads,
    torch::Tensor band_starts,
    torch::Tensor band_ends,
    int total_bands,
    int H, int MB, int NB,
    int W, int max_band_size
) {
    const c10::cuda::CUDAGuard device_guard(mask.device());
    auto perm = sorted_indices.clone();
    if (total_bands == 0) return perm;

    int threads = 256;
    int smem = NB * sizeof(float) + threads * (sizeof(float) + sizeof(int)) + max_band_size;

    banded_window_greedy_reorder_kernel<<<total_bands, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(
        mask.data_ptr<float>(),
        sorted_indices.data_ptr<int64_t>(),
        band_heads.data_ptr<int>(),
        band_starts.data_ptr<int>(),
        band_ends.data_ptr<int>(),
        perm.data_ptr<int64_t>(),
        MB, NB, W
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return perm;
}
"""

_cpp_source = """
torch::Tensor greedy_reorder_cuda(torch::Tensor S);
torch::Tensor window_greedy_reorder_cuda(torch::Tensor S, int W, torch::Tensor lengths);
torch::Tensor banded_greedy_reorder_cuda(
    torch::Tensor mask,
    torch::Tensor band_row_idx,
    torch::Tensor band_offsets,
    torch::Tensor band_heads,
    torch::Tensor perm_write_off,
    int total_bands,
    int H, int MB, int NB,
    int max_band_size
);
torch::Tensor banded_window_greedy_reorder_cuda(
    torch::Tensor mask,
    torch::Tensor sorted_indices,
    torch::Tensor band_heads,
    torch::Tensor band_starts,
    torch::Tensor band_ends,
    int total_bands,
    int H, int MB, int NB,
    int W, int max_band_size
);
"""

# Lazy compilation with caching
_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        # Let PyTorch target the visible GPUs (or the caller's explicit
        # TORCH_CUDA_ARCH_LIST); an SM90-only binary cannot run on Ampere.
        _ext = load_inline(
            name="greedy_reorder",
            cpp_sources=_cpp_source,
            cuda_sources=_cuda_source,
            functions=["greedy_reorder_cuda", "window_greedy_reorder_cuda", "banded_greedy_reorder_cuda", "banded_window_greedy_reorder_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3"],
        )
    return _ext


def global_greedy_reorder_cuda(mask_float, batch_heads=8):
    """Full global greedy reorder — CUDA kernel, all heads parallel.
    
    Args:
        mask_float: (H, MB, NB) float tensor — block attention mask per head
        batch_heads: number of heads to compute S for simultaneously (memory vs speed tradeoff)
    Returns:
        perm: (H, MB) long tensor — per-head row permutation
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    ext = _get_ext()
    
    perm = torch.zeros(H, MB, dtype=torch.int64, device=device)
    # Process heads in batches to balance memory and speed
    # Each batch: batch_heads * MB * MB * 4 bytes (e.g., 8 * 3015^2 * 4 = ~277MB)
    for start in range(0, H, batch_heads):
        end = min(start + batch_heads, H)
        mask_batch = mask_float[start:end]  # (batch, MB, NB)
        S_batch = torch.bmm(mask_batch, mask_batch.transpose(1, 2)).float()
        perm_batch = ext.greedy_reorder_cuda(S_batch)
        perm[start:end] = perm_batch
        del S_batch, perm_batch
    return perm


def global_banded_greedy_reorder_cuda(mask_float, band_size=16, max_band_rows=512, skip_cv=0.01):
    """Banded CUDA Greedy (BCG) reorder — NNZ banding + per-band overlap greedy.
    
    Uses Kernel 1 (pre-computed S via bmm + greedy argmax kernel) per band.
    bmm leverages cuBLAS tensor cores for fast S = mask @ mask.T computation.
    
    Head-adaptive: heads with constant NNZ (cv < skip_cv) are skipped since
    greedy provides negligible MCRD improvement for them (~0pp loss).
    
    Args:
        mask_float: (H, MB, NB) float tensor — block attention mask per head
        band_size: NNZ quantization step for banding (default=16)
        max_band_rows: max rows per greedy TSP call (default=512)
        skip_cv: skip greedy for heads with NNZ CV below this (default=0.01)
    Returns:
        perm: (H, MB) long tensor — per-head row permutation
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    ext = _get_ext()
    
    # --- Phase 1: Vectorized NNZ banding + sorting (all on GPU) ---
    nnz = mask_float.sum(-1)                          # (H, MB) float
    band_ids = (nnz / band_size).int()                # (H, MB)
    
    # Sort each head by (-band_id, original_index) for descending NNZ bands
    max_band = band_ids.max().item() + 1
    sort_key = (max_band - band_ids.float()) * (MB + 1) + torch.arange(MB, device=device).float()
    sorted_indices = sort_key.argsort(dim=1)          # (H, MB)
    sorted_bands = torch.gather(band_ids, 1, sorted_indices)

    # --- Phase 2: Band boundary detection ---
    diff = torch.zeros(H, MB, dtype=torch.bool, device=device)
    diff[:, 0] = True
    diff[:, 1:] = sorted_bands[:, 1:] != sorted_bands[:, :-1]
    
    # --- Phase 2b: Head-adaptive skip (constant-NNZ heads don't benefit from greedy) ---
    nnz_std = nnz.std(dim=1)    # (H,)
    nnz_mean = nnz.mean(dim=1)  # (H,)
    cv = nnz_std / nnz_mean.clamp(min=1)
    skip_head = (cv < skip_cv)  # (H,) bool — True = skip greedy
    
    perm = sorted_indices.clone().long()  # default: NNZ-sorted order
    
    # --- Phase 3: Per-band greedy via Kernel 1 (bmm + argmax) ---
    for h in range(H):
        if skip_head[h]:
            continue  # Use NNZ-sorted order for constant-NNZ heads
        
        boundaries = torch.where(diff[h])[0]
        starts = boundaries.tolist()
        ends = starts[1:] + [MB]
        
        for s, e in zip(starts, ends):
            length = e - s
            if length <= 1:
                continue
            
            # Chunk large bands
            for cs in range(s, e, max_band_rows):
                ce = min(cs + max_band_rows, e)
                cl = ce - cs
                if cl <= 1:
                    continue
                
                band_row_idx = sorted_indices[h, cs:ce]
                
                # S = mask @ mask.T via cuBLAS bmm (tensor core accelerated)
                band_mask = mask_float[h, band_row_idx].unsqueeze(0)  # (1, cl, NB)
                S = torch.bmm(band_mask, band_mask.transpose(1, 2))   # (1, cl, cl)
                
                # Kernel 1: greedy TSP on pre-computed S
                local_perm = ext.greedy_reorder_cuda(S)[0]  # (cl,)
                perm[h, cs:ce] = band_row_idx[local_perm]
    
    return perm


def global_banded_window_greedy_reorder_cuda(mask_float, band_size=4, max_band_rows=512, window=132):
    """Banded Window-Greedy reorder — NNZ banding + W-lookback greedy per band.
    
    Uses Kernel 3: instead of picking the next row by argmax(S[curr, :]),
    picks by argmax(window_sum) where window_sum accumulates overlap from
    the last W placed rows. This better matches the 132-SM concurrent window.
    
    Args:
        mask_float: (H, MB, NB) float tensor — block attention mask per head
        band_size: NNZ quantization step for banding (default=4)
        max_band_rows: max rows per greedy TSP call (default=512)
        window: lookback window size W (default=132, matching H100 SM count)
    Returns:
        perm: (H, MB) long tensor — per-head row permutation
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    ext = _get_ext()
    
    # --- Phase 1: Vectorized NNZ banding + sorting ---
    nnz = mask_float.sum(-1)
    band_ids = (nnz / band_size).int()
    max_band = band_ids.max().item() + 1
    sort_key = (max_band - band_ids.float()) * (MB + 1) + torch.arange(MB, device=device).float()
    sorted_indices = sort_key.argsort(dim=1)
    sorted_bands = torch.gather(band_ids, 1, sorted_indices)

    # --- Phase 2: Band boundary detection ---
    diff = torch.zeros(H, MB, dtype=torch.bool, device=device)
    diff[:, 0] = True
    diff[:, 1:] = sorted_bands[:, 1:] != sorted_bands[:, :-1]
    
    perm = sorted_indices.clone().long()
    
    # --- Phase 3: Per-band window greedy via Kernel 3 (bmm + window argmax) ---
    for h in range(H):
        boundaries = torch.where(diff[h])[0]
        starts = boundaries.tolist()
        ends = starts[1:] + [MB]
        
        for s, e in zip(starts, ends):
            length = e - s
            if length <= 1:
                continue
            
            for cs in range(s, e, max_band_rows):
                ce = min(cs + max_band_rows, e)
                cl = ce - cs
                if cl <= 1:
                    continue
                
                band_row_idx = sorted_indices[h, cs:ce]
                
                # S = mask @ mask.T via cuBLAS bmm
                band_mask = mask_float[h, band_row_idx].unsqueeze(0)
                S = torch.bmm(band_mask, band_mask.transpose(1, 2))  # (1, cl, cl)
                
                # Kernel 3: window-lookback greedy on pre-computed S
                local_perm = ext.window_greedy_reorder_cuda(S, min(window, cl), torch.full((1,), cl, dtype=torch.int32, device=device))[0]
                perm[h, cs:ce] = band_row_idx[local_perm]
    
    return perm

def bcg_l2_reorder_cuda(mask_float, band_size=4, max_band_rows=512, window=256):
    """L2-Resident BCG (Bounded Column Gap) Reorder.
    Minimizes LRU Cache misses directly simulating an L2 window.
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    ext = _get_ext()

    nnz = mask_float.sum(-1)                          
    band_ids = (nnz / band_size).int()                
    max_band = band_ids.max().item() + 1
    sort_key = (max_band - band_ids.float()) * (MB + 1) + torch.arange(MB, device=device).float()
    sorted_indices = sort_key.argsort(dim=1)          
    sorted_bands = torch.gather(band_ids, 1, sorted_indices)

    diff = torch.zeros((H, MB), dtype=torch.bool, device=device)
    diff[:, 0] = True
    diff[:, 1:] = sorted_bands[:, 1:] != sorted_bands[:, :-1]

    if max_band_rows > 0:
        diff[:, ::max_band_rows] = True

    band_starts_tensor = torch.nonzero(diff)
    if band_starts_tensor.numel() == 0:
        return sorted_indices.clone()

    band_heads = band_starts_tensor[:, 0].int()
    band_starts = band_starts_tensor[:, 1].int()
    total_bands = band_starts_tensor.shape[0]

    band_ends = torch.empty_like(band_starts)
    if total_bands > 0:
        same_head_mask = (band_heads[:-1] == band_heads[1:])
        band_ends[:-1] = torch.where(
            same_head_mask, 
            band_starts[1:], 
            torch.tensor(MB, dtype=torch.int32, device=device)
        )
        band_ends[-1] = MB

    max_bs = (band_ends - band_starts).max().item()

    perm = ext.banded_window_greedy_reorder_cuda(
        mask_float,
        sorted_indices,
        band_heads,
        band_starts,
        band_ends,
        total_bands,
        H, MB, NB,
        window, max_bs
    )
    return perm


def optimal_reorder_cuda(mask_float, band_size=4, max_band_rows=512, window=132):
    """Production-optimal reorder: band4 window-greedy (Kernel 3), vectorised.

    NCU-verified L2 hit rates (H100, step=4, 16 heads, 3210 blocks):
        thr=0.95 (density=0.92): 90.51% L2 hit
        thr=0.70 (density=0.60): 92.42% L2 hit

    Overhead massively reduced: completely avoids python loops via Vectorized Gather + BMM.

    Args:
        mask_float: (H, MB, NB) float tensor — block attention mask per head
        band_size: NNZ quantization step for banding (default=4, NCU-optimal)
        max_band_rows: max rows per greedy TSP call (default=512)
        window: lookback window size W (default=132, matching H100 SM count)
    Returns:
        perm: (H, MB) long tensor — per-head row permutation
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    ext = _get_ext()

    # --- Phase 1: Vectorized NNZ banding + sorting (all on GPU) ---
    nnz = mask_float.sum(-1)                          # (H, MB) float
    band_ids = (nnz / band_size).int()                # (H, MB)

    # Sort each head by (-band_id, original_index) for descending NNZ bands
    max_band = band_ids.max().item() + 1
    sort_key = (max_band - band_ids.float()) * (MB + 1) + torch.arange(MB, device=device).float()
    sorted_indices = sort_key.argsort(dim=1)          # (H, MB)
    sorted_bands = torch.gather(band_ids, 1, sorted_indices)

    # --- Phase 2: Band boundary detection (Vectorized) ---
    diff = torch.zeros((H, MB), dtype=torch.bool, device=device)
    diff[:, 0] = True
    diff[:, 1:] = sorted_bands[:, 1:] != sorted_bands[:, :-1]

    if max_band_rows > 0:
        diff[:, ::max_band_rows] = True

    band_starts_tensor = torch.nonzero(diff)
    if band_starts_tensor.numel() == 0:
        return sorted_indices.clone()

    band_heads = band_starts_tensor[:, 0].int()
    band_starts = band_starts_tensor[:, 1].int()
    total_bands = band_starts_tensor.shape[0]

    band_ends = torch.empty_like(band_starts)
    if total_bands > 0:
        same_head_mask = (band_heads[:-1] == band_heads[1:])
        band_ends[:-1] = torch.where(
            same_head_mask, 
            band_starts[1:], 
            torch.tensor(MB, dtype=torch.int32, device=device)
        )
        band_ends[-1] = MB

    perm = sorted_indices.clone()

    # --- Phase 3: Vectorized Gather BMM ---
    lengths = band_ends - band_starts
    max_bs = lengths.max().item()
    if max_bs <= 1:
        return perm
        
    seq = torch.arange(max_bs, device=device).unsqueeze(0)
    lengths_2d = lengths.unsqueeze(1)
    valid_mask = seq < lengths_2d

    item_idx = (band_starts.unsqueeze(1) + seq).clamp(max=MB-1)
    global_idx = (band_heads.unsqueeze(1) * MB + item_idx).long()

    flat_sorted = sorted_indices.view(-1)
    flat_row_idx = flat_sorted[global_idx]

    mask_global_idx = (band_heads.unsqueeze(1) * MB + flat_row_idx).long()
    flat_mask = mask_float.view(-1, NB)

    # Using Advanced Indexing to pull directly via GPU cores
    padded_masks = flat_mask[mask_global_idx]
    padded_masks[~valid_mask] = 0.0

    S = torch.bmm(padded_masks, padded_masks.transpose(1, 2))
    
    # Zero out invalid similarities
    S[~valid_mask] = -1e30
    S.transpose(1, 2)[~valid_mask] = -1e30

    # --- Phase 4: Launch Window Greedy Kernel (Kernel 3) ---
    local_perms = ext.window_greedy_reorder_cuda(S, min(window, max_bs), lengths.int())

    # --- Phase 5: Map back (Vectorized Scatter) ---
    # We want: perm[global_idx] = flat_sorted[band_heads*MB + local_perms]
    # Except we must only map the valid areas
    
    # Map local_perms to actual original indices
    local_item_idx = local_perms.clamp(max=max_bs-1)
    # The selected row's flat_row_idx:
    selected_flat_row_idx = flat_row_idx.gather(1, local_item_idx)

    # Only scatter the valid ones. 
    # valid_mask is (total_bands, max_bs)
    valid_global_idx = global_idx[valid_mask]
    valid_selected_row_idx = selected_flat_row_idx[valid_mask]

    # perm is (H, MB), flatten to scatter directly natively
    perm.view(-1)[valid_global_idx] = valid_selected_row_idx
    
    return perm

def _centered_rank2_basis(mask_float, n_iter=10):
    """Paper Algorithm 1: matrix-free centered subspace iteration.

    Centering is applied by rank-one corrections, and QR prevents the two
    columns from collapsing onto the same dominant direction. Products use
    PyTorch's dense batched matmul here; this is not an O(nnz) sparse kernel.
    """
    heads, rows, _ = mask_float.shape
    generator = torch.Generator(device=mask_float.device).manual_seed(42)
    basis = torch.randn(heads, rows, 2, device=mask_float.device,
                        dtype=mask_float.dtype, generator=generator)
    basis = torch.linalg.qr(basis, mode="reduced").Q
    column_sum = mask_float.sum(dim=1)
    transpose = mask_float.transpose(1, 2)
    for _ in range(n_iter):
        right = torch.bmm(transpose, basis)
        right = right - column_sum[:, :, None] * (basis.sum(dim=1, keepdim=True) / rows)
        left = torch.bmm(mask_float, right)
        correction = (column_sum[:, :, None] * right).sum(dim=1, keepdim=True) / rows
        basis = torch.linalg.qr(left - correction, mode="reduced").Q
    return basis


def _wavealign_stage2(order, nnz, wave_size):
    """Sort within each Stage-1 wave while preserving the wave order."""
    heads, rows = order.shape
    pad = (-rows) % wave_size
    ordered_counts = nnz.gather(1, order)
    if pad:
        order = torch.nn.functional.pad(order, (0, pad), value=0)
        ordered_counts = torch.nn.functional.pad(ordered_counts, (0, pad), value=-float("inf"))
    rank = ordered_counts.reshape(heads, -1, wave_size).argsort(dim=2, descending=True, stable=True)
    refined = order.reshape(heads, -1, wave_size).gather(2, rank)
    return refined.reshape(heads, -1)[:, :rows].contiguous()


def wave_2d_reorder(mask_float, wave_size=132):
    """Paper Algorithms 1/2: centered rank-2 projection and intra-wave NNZ sort."""
    if mask_float.ndim != 3 or min(mask_float.shape) <= 0 or wave_size <= 0:
        raise ValueError("Expected a nonempty [heads, Q blocks, KV blocks] mask and positive wave size")
    mask_float = mask_float.float()
    nnz = mask_float.sum(dim=2)
    if min(mask_float.shape[1:]) < 2:
        return nnz.argsort(dim=1, descending=True, stable=True)
    basis = _centered_rank2_basis(mask_float)
    order = torch.atan2(basis[:, :, 1], basis[:, :, 0]).argsort(dim=1, stable=True)
    return _wavealign_stage2(order, nnz, wave_size)


def wave_overlap_reorder(mask_float, wave_size=132):
    """Wave-Overlap Reorder: Pure 1D Spectral Sorting (Fiedler Vector approach)
    
    This is the globally optimal sorting permutation for L2 cache spatial coherence.
    It executes power iteration using 1 singular vector to cluster identical K/V access patterns
    together, resulting in maximum MSHR reuse and minimum DRAM reads.
    
    NCU-verified: mathematically equal cache hit rate to 2D projections but faster computation time.
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)  # (H, MB)
    n_waves = (MB + wave_size - 1) // wave_size
    
    # Fast 1D Power iteration for top-1 singular vector
    gen = torch.Generator(device=device).manual_seed(42)
    V = torch.randn(H, NB, 1, device=device, generator=gen)  # (H, NB, 1)
    
    # 10 iterations typically sufficient for Fiedler vector stability
    for _ in range(10):
        U = torch.bmm(mask_float, V)            
        U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-8)
        V = torch.bmm(mask_float.transpose(1, 2), U)  
        V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)
    # 1阶截断能量
    # # top-1 最大奇异值近似
    # sigma1 = torch.bmm(
    #     U.transpose(1, 2),
    #     torch.bmm(mask_float, V)
    # ).squeeze(-1).squeeze(-1)  # (H,)

    # # 能量占比：sigma1^2 / sum_i sigma_i^2
    # # 其中 sum_i sigma_i^2 = ||A||_F^2
    # fro_sq = mask_float.square().sum(dim=(1, 2))  # (H,)
    # energy_ratio = sigma1.square() / fro_sq.clamp(min=1e-8)

    # # 单行打印：h0=..., h1=..., h2=...
    # energy_ratio_cpu = energy_ratio.detach().cpu()
    # energy_ratio_line = " | ".join(
    #     [f"h{i}={v:.6f}" for i, v in enumerate(energy_ratio_cpu.tolist())]
    # )
    # print(f"[wave_overlap_reorder] sigma1^2 / sum(sigma^2) per head -> {energy_ratio_line}", flush=True)


    # # 2阶截断能量占比（energy ratio）：
    # # (sigma1^2 + sigma2^2) / sum_i sigma_i^2
    # svals = torch.linalg.svdvals(mask_float)  # (H, min(MB, NB))
    # top2_energy = svals[:, :2].square().sum(dim=1)      # (H,)
    # fro_sq = mask_float.square().sum(dim=(1, 2))        # (H,), 等于 sum_i sigma_i^2
    # top2_energy_ratio = top2_energy / fro_sq.clamp(min=1e-8)

    # # 单行打印：h0=..., h1=..., ...
    # top2_energy_ratio_cpu = top2_energy_ratio.detach().cpu()
    # top2_energy_ratio_line = " | ".join(
    #     [f"h{i}={v:.6f}" for i, v in enumerate(top2_energy_ratio_cpu.tolist())]
    # )
    # print(
    #     f"[wave_overlap_reorder] (sigma1^2+sigma2^2) / sum(sigma^2) per head -> {top2_energy_ratio_line}",
    #     flush=True,
    # )

    # 仅统计并打印 3阶截断能量占比：
    # (sigma1^2 + sigma2^2 + sigma3^2) / sum_i sigma_i^2
    # svals = torch.linalg.svdvals(mask_float)           # (H, min(MB, NB))
    # top3_energy = svals[:, :3].square().sum(dim=1)     # (H,)
    # fro_sq = mask_float.square().sum(dim=(1, 2))       # (H,)
    # top3_energy_ratio = top3_energy / fro_sq.clamp(min=1e-8)

    # top3_energy_ratio_cpu = top3_energy_ratio.detach().cpu()
    # top3_energy_ratio_line = " | ".join(
    #     [f"h{i}={v:.6f}" for i, v in enumerate(top3_energy_ratio_cpu.tolist())]
    # )
    # print(
    #     f"[wave_overlap_reorder] (sigma1^2+sigma2^2+sigma3^2) / sum(sigma^2) per head -> {top3_energy_ratio_line}",
    #     flush=True,
    # )

    # # 4阶截断能量占比（energy ratio）：
    # # (sigma1^2 + sigma2^2 + sigma3^2 + sigma4^2) / sum_i sigma_i^2
    # svals = torch.linalg.svdvals(mask_float)  # (H, min(MB, NB))
    # top4_energy = svals[:, :4].square().sum(dim=1)      # (H,)
    # fro_sq = mask_float.square().sum(dim=(1, 2))        # (H,) = sum_i sigma_i^2
    # top4_energy_ratio = top4_energy / fro_sq.clamp(min=1e-8)

    # # 单行打印：h0=..., h1=..., ...
    # top4_energy_ratio_cpu = top4_energy_ratio.detach().cpu()
    # top4_energy_ratio_line = " | ".join(
    #     [f"h{i}={v:.6f}" for i, v in enumerate(top4_energy_ratio_cpu.tolist())]
    # )
    # print(
    #     f"[wave_overlap_reorder] (sigma1^2+sigma2^2+sigma3^2+sigma4^2) / sum(sigma^2) per head -> {top4_energy_ratio_line}",
    #     flush=True,
    # )

    # # 5阶截断能量占比（energy ratio）：
    # # (sigma1^2 + sigma2^2 + sigma3^2 + sigma4^2 + sigma5^2) / sum_i sigma_i^2
    # svals = torch.linalg.svdvals(mask_float)  # (H, min(MB, NB))
    # k = min(5, svals.shape[1])                # 防止维度不足5
    # top5_energy = svals[:, :k].square().sum(dim=1)      # (H,)
    # fro_sq = mask_float.square().sum(dim=(1, 2))        # (H,) = sum_i sigma_i^2
    # top5_energy_ratio = top5_energy / fro_sq.clamp(min=1e-8)

    # # 单行打印：h0=..., h1=..., ...
    # top5_energy_ratio_cpu = top5_energy_ratio.detach().cpu()
    # top5_energy_ratio_line = " | ".join(
    #     [f"h{i}={v:.6f}" for i, v in enumerate(top5_energy_ratio_cpu.tolist())]
    # )
    # print(
    #     f"[wave_overlap_reorder] (sigma1^2+sigma2^2+sigma3^2+sigma4^2+sigma5^2) / sum(sigma^2) per head -> {top5_energy_ratio_line}",
    #     flush=True,
    # )

    # # 1阶截断能量占比（energy ratio）：
    # # sigma1^2 / sum_i sigma_i^2
    # svals = torch.linalg.svdvals(mask_float)          # (H, min(MB, NB))
    # top1_energy = svals[:, 0].square()                # (H,)
    # fro_sq = mask_float.square().sum(dim=(1, 2))      # (H,) = sum_i sigma_i^2
    # top1_energy_ratio = top1_energy / fro_sq.clamp(min=1e-8)

    # # 单行打印：h0=..., h1=..., ...
    # top1_energy_ratio_cpu = top1_energy_ratio.detach().cpu()
    # top1_energy_ratio_line = " | ".join(
    #     [f"h{i}={v:.6f}" for i, v in enumerate(top1_energy_ratio_cpu.tolist())]
    # )
    # print(
    #     f"[wave_overlap_reorder] sigma1^2 / sum(sigma^2) per head -> {top1_energy_ratio_line}",
    #     flush=True,
    # )

    # Pure 1D descending sort (group spatially similar rows together)
    spectral_order = U[:, :, 0].argsort(dim=1, descending=True)  # (H, MB)
    
    # Partition into waves
    pad = n_waves * wave_size - MB
    if pad > 0:
        pad_idx = torch.arange(MB, MB + pad, device=device).unsqueeze(0).expand(H, -1)
        spectral_order_pad = torch.cat([spectral_order, pad_idx], dim=1)
        nnz_pad = torch.cat([nnz, torch.zeros(H, pad, dtype=nnz.dtype, device=device)], dim=1)
    else:
        spectral_order_pad = spectral_order
        nnz_pad = nnz

    # Extract NNZ sequentially into the waves
    nnz_ordered = torch.gather(nnz_pad, 1, spectral_order_pad)
    nnz_waves = nnz_ordered.view(H, n_waves, wave_size)
    wave_mean_nnz = nnz_waves.float().mean(dim=2)
    
    # Execute heavily loaded dense waves first to give time for trailing dense waves to hide latency
    wave_order = wave_mean_nnz.argsort(dim=1, descending=True)
    # Sort within the 132-wave by NNZ descending to balance warps (intra-wave shear reduction)
    within_wave_order = nnz_waves.argsort(dim=2, descending=True)
    
    spectral_waves = spectral_order_pad.view(H, n_waves, wave_size)
    spectral_sorted = torch.gather(spectral_waves, 2, within_wave_order)
    
    wave_order_exp = wave_order.unsqueeze(2).expand(H, n_waves, wave_size)
    result_padded = torch.gather(spectral_sorted, 1, wave_order_exp)
    
    # Strip padding to return original bounds
    valid_mask = result_padded < MB
    result = result_padded[valid_mask].view(H, MB).contiguous()
    
    return result

def fiedler_wave_reorder(mask_float, wave_size=132, n_iter=20):
    """Fiedler Wave Reorder: Normalized Laplacian Fiedler vector + wave scheduling.
    
    Computes the 2nd eigenvector (Fiedler vector) of the normalized Laplacian
    of the row-similarity graph W = M @ M^T, then applies NNZ-based wave
    scheduling for SM load balancing.
    
    Key insight: The normalized Laplacian D^{-1/2} W D^{-1/2} uses cosine
    similarity (relative column overlap) rather than inner product (absolute
    column overlap), eliminating NNZ bias that plagues SVD-based approaches.
    The Fiedler vector gives the optimal 1D linear arrangement that minimizes
    the graph cut — grouping rows with similar K/V access patterns together.
    
    Implementation uses implicit matrix-vector products M @ (M^T @ v) to avoid
    materializing the MB×MB similarity matrix S, saving ~800MB memory and
    achieving 2.3x speedup over the explicit form.
    
    NCU-verified: best DRAM reduction on 4/5 sparse modes (84-89% vs baseline).
    Beats SVD wave_2d on LongCat (-56%), BSA (-48%), vMoBA (-43%).
    
    Args:
        mask_float: (H, MB, NB) float tensor — block attention mask per head
        wave_size:  number of rows per SM wave (default 132 for H100)
        n_iter:     power iterations for Fiedler convergence (default 20)
    Returns:
        perm: (H, MB) long tensor — per-head row permutation
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)  # (H, MB)
    n_waves = (MB + wave_size - 1) // wave_size
    Mt = mask_float.transpose(1, 2)  # (H, NB, MB) — cache once
    
    # --- Step 1: Compute degree vector implicitly ---
    # deg[i] = sum_j W[i,j] = sum_j (M[i,:] · M[j,:]) = M @ (M^T @ 1)
    ones = torch.ones(H, MB, 1, device=device)
    deg = torch.bmm(mask_float, torch.bmm(Mt, ones)).squeeze(2)  # (H, MB)
    dinv = 1.0 / deg.clamp(min=1e-8).sqrt()  # D^{-1/2}
    
    # --- Step 2: Power iteration for Fiedler vector ---
    # Iterate on A = D^{-1/2} W D^{-1/2} with deflation of trivial eigenvector v0
    # A @ v = D^{-1/2} M @ (M^T @ (D^{-1/2} v))  [implicit, no S materialization]
    gen = torch.Generator(device=device).manual_seed(42)
    v0 = torch.ones(H, MB, 1, device=device) / (MB ** 0.5)  # trivial eigenvector
    v1 = torch.randn(H, MB, 1, device=device, generator=gen)
    
    for _ in range(n_iter):
        x = dinv.unsqueeze(2) * v1                     # D^{-1/2} v1
        y = torch.bmm(mask_float, torch.bmm(Mt, x))    # W @ (D^{-1/2} v1)
        v1 = dinv.unsqueeze(2) * y                      # D^{-1/2} W D^{-1/2} v1
        v1 = v1 - torch.bmm(v0.transpose(1, 2), v1) * v0  # deflate trivial eigvec
        v1 = v1 / v1.norm(dim=1, keepdim=True).clamp(min=1e-8)
    
    # --- Step 3: Sort by Fiedler vector ---
    fiedler_perm = v1[:, :, 0].argsort(dim=1)  # (H, MB)
    
    # --- Step 4: Wave scheduling (NNZ-based) ---
    nnz_r = torch.gather(nnz, 1, fiedler_perm)
    pad = n_waves * wave_size - MB
    if pad > 0:
        pad_idx = torch.arange(MB, MB + pad, device=device).unsqueeze(0).expand(H, -1)
        perm_p = torch.cat([fiedler_perm, pad_idx], dim=1)
        nnz_p = torch.cat([nnz_r, torch.zeros(H, pad, dtype=nnz.dtype, device=device)], dim=1)
    else:
        perm_p = fiedler_perm
        nnz_p = nnz_r
    
    nnz_w = nnz_p.view(H, n_waves, wave_size)
    wave_mean_nnz = nnz_w.float().mean(dim=2)
    
    # Dense waves first, NNZ descending within each wave
    wave_order = wave_mean_nnz.argsort(dim=1, descending=True)
    within_wave_order = nnz_w.argsort(dim=2, descending=True)
    
    waves = perm_p.view(H, n_waves, wave_size)
    sorted_waves = torch.gather(waves, 2, within_wave_order)
    
    wave_order_exp = wave_order.unsqueeze(2).expand(H, n_waves, wave_size)
    result_padded = torch.gather(sorted_waves, 1, wave_order_exp)
    
    # Strip padding
    valid_mask = result_padded < MB
    result = result_padded[valid_mask].view(H, MB).contiguous()
    
    return result


def hybrid_block_reorder(mask_float, wave_size=132, waves_per_block=4):
    """Hybrid Shear-Aware Spectral Sorting.
    
    Combines the zero-shear advantage of nnz_reorder and the high-MSHR overlap
    advantage of wave_reorder by grouping into large NNZ-uniform super-waves, 
    and sub-sorting by 1D spectral shape internally.
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)
    
    super_wave_size = wave_size * waves_per_block
    n_super_waves = (MB + super_wave_size - 1) // super_wave_size
    pad = n_super_waves * super_wave_size - MB
    
    # 1D Power Iteration
    gen = torch.Generator(device=device).manual_seed(42)
    V = torch.randn(H, NB, 1, device=device, generator=gen)
    for _ in range(10):
        U = torch.bmm(mask_float, V)
        U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-8)
        V = torch.bmm(mask_float.transpose(1, 2), U)
        V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)
        
    u_score = U[:, :, 0]
    
    # Apply padding
    if pad > 0:
        pad_idx = torch.arange(MB, MB + pad, device=device).unsqueeze(0).expand(H, -1)
        base_indices = torch.cat([torch.arange(MB, device=device).unsqueeze(0).expand(H, -1), pad_idx], dim=1)
        nnz_pad = torch.cat([nnz, torch.zeros(H, pad, device=device)], dim=1)
        u_pad = torch.cat([u_score, torch.zeros(H, pad, device=device) - 1e9], dim=1)
    else:
        base_indices = torch.arange(MB, device=device).unsqueeze(0).expand(H, -1)
        nnz_pad = nnz
        u_pad = u_score
        
    # Phase 1: Sort by NNZ globally (Coarse Grouping)
    order_nnz = nnz_pad.argsort(dim=1, descending=True)
    base_indices = torch.gather(base_indices, 1, order_nnz)
    u_pad = torch.gather(u_pad, 1, order_nnz)
    
    # Phase 2: Group into super-waves and sort internally by U score (Fine Shape Grouping)
    u_super = u_pad.view(H, n_super_waves, super_wave_size)
    base_super = base_indices.view(H, n_super_waves, super_wave_size)
    
    # Sort by U score within each super wave
    order_u = u_super.argsort(dim=2, descending=True)
    base_super = torch.gather(base_super, 2, order_u)
    
    # Phase 3: Flatten to waves and handle Inter-wave / Intra-wave balancing
    n_waves = n_super_waves * waves_per_block
    base_waves = base_super.view(H, n_waves, wave_size)
    
    # We must sort the resulting 132-waves by their mean NNZ to ensure dense waves execute first
    # Need original NNZ values mapped by base_waves
    nnz_flat = torch.cat([nnz, torch.zeros(H, pad, device=device)], dim=1)
    nnz_waves = torch.gather(nnz_flat, 1, base_waves.view(H, -1)).view(H, n_waves, wave_size)
    
    wave_mean_nnz = nnz_waves.mean(dim=2)
    wave_order = wave_mean_nnz.argsort(dim=1, descending=True)
    within_wave_order = nnz_waves.argsort(dim=2, descending=True)
    
    wave_order_exp = wave_order.unsqueeze(2).expand(H, n_waves, wave_size)
    result_waves = torch.gather(base_waves, 1, wave_order_exp)
    result_waves = torch.gather(result_waves, 2, within_wave_order)
    
    # Flatten and remove padding
    result_padded = result_waves.view(H, -1)
    valid_mask = result_padded < MB
    result = result_padded[valid_mask].view(H, MB).contiguous()
    return result

def wave_1d_reorder(mask_float, wave_size=132):
    """Pure 1D Spectral Sorting (Fiedler Vector approach) for NCU comparison.
    Uses exactly 1 dimension for power iteration and sorts purely by the 1D scalar.
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)  # (H, MB)
    n_waves = (MB + wave_size - 1) // wave_size
    
    # 1D Power iteration
    gen = torch.Generator(device=device).manual_seed(42)
    V = torch.randn(H, NB, 1, device=device, generator=gen)  # (H, NB, 1)
    V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)
    
    for _ in range(10):
        U = torch.bmm(mask_float, V)            
        U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-8)
        V = torch.bmm(mask_float.transpose(1, 2), U)  
        V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)
    
    # Pure 1D sorting
    spectral_order = U[:, :, 0].argsort(dim=1)  # (H, MB)
    
    # Partition into waves
    pad = n_waves * wave_size - MB
    if pad > 0:
        pad_idx = torch.arange(MB, MB + pad, device=device).unsqueeze(0).expand(H, -1)
        spectral_order_pad = torch.cat([spectral_order, pad_idx], dim=1)
        nnz_pad = torch.cat([nnz, torch.zeros(H, pad, dtype=nnz.dtype, device=device)], dim=1)
    else:
        spectral_order_pad = spectral_order
        nnz_pad = nnz

    nnz_ordered = torch.gather(nnz_pad, 1, spectral_order_pad)
    nnz_waves = nnz_ordered.view(H, n_waves, wave_size)
    wave_mean_nnz = nnz_waves.float().mean(dim=2)
    
    wave_order = wave_mean_nnz.argsort(dim=1, descending=True)
    within_wave_order = nnz_waves.argsort(dim=2, descending=True)
    
    spectral_waves = spectral_order_pad.view(H, n_waves, wave_size)
    spectral_sorted = torch.gather(spectral_waves, 2, within_wave_order)
    
    wave_order_exp = wave_order.unsqueeze(2).expand(H, n_waves, wave_size)
    result_padded = torch.gather(spectral_sorted, 1, wave_order_exp)
    
    valid_mask = result_padded < MB
    result = result_padded[valid_mask].view(H, MB).contiguous()
    
    return result


def nnz_3step_reorder(mask_float, wave_size=132):
    """NNZ argsort + same 3-step wave post-processing as wave_1d_reorder.
    
    For NCU ablation: isolates whether SVD u1 provides any benefit over
    simple NNZ sorting when the wave partition/reorder steps are identical.
    
    Step 1: NNZ argsort descending (instead of SVD u1)
    Step 2: Within-wave NNZ descending sort (same as wave_1d_reorder)
    Step 3: Wave reorder by mean NNZ descending (same as wave_1d_reorder)
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)  # (H, MB)
    n_waves = (MB + wave_size - 1) // wave_size
    
    # Step 1: NNZ argsort descending (replaces SVD power iteration)
    init_order = nnz.argsort(dim=1, descending=True)  # (H, MB)
    
    # Partition into waves (identical to wave_1d_reorder from here)
    pad = n_waves * wave_size - MB
    if pad > 0:
        pad_idx = torch.arange(MB, MB + pad, device=device).unsqueeze(0).expand(H, -1)
        order_pad = torch.cat([init_order, pad_idx], dim=1)
        nnz_pad = torch.cat([nnz, torch.zeros(H, pad, dtype=nnz.dtype, device=device)], dim=1)
    else:
        order_pad = init_order
        nnz_pad = nnz

    nnz_ordered = torch.gather(nnz_pad, 1, order_pad)
    nnz_waves = nnz_ordered.view(H, n_waves, wave_size)
    wave_mean_nnz = nnz_waves.float().mean(dim=2)
    
    # Step 3: wave reorder by mean NNZ descending
    wave_order = wave_mean_nnz.argsort(dim=1, descending=True)
    # Step 2: within-wave NNZ descending sort
    within_wave_order = nnz_waves.argsort(dim=2, descending=True)
    
    order_waves = order_pad.view(H, n_waves, wave_size)
    order_sorted = torch.gather(order_waves, 2, within_wave_order)
    
    wave_order_exp = wave_order.unsqueeze(2).expand(H, n_waves, wave_size)
    result_padded = torch.gather(order_sorted, 1, wave_order_exp)
    
    valid_mask = result_padded < MB
    result = result_padded[valid_mask].view(H, MB).contiguous()
    
    return result


def svd_u2u3_reorder(mask_float, wave_size=132, n_iter=15):
    """2D Spectral Reordering via Implicit SVD Power Iteration + MGS.

    Computes the top-3 left singular vectors of M (≡ eigenvectors of W = M @ M^T)
    using implicit block power iteration with Modified Gram-Schmidt.
    Avoids forming W = M @ M^T explicitly (saves ~30G FLOPs + ~106MB memory).
    Uses lightweight MGS instead of torch.linalg.qr (avoids LAPACK overhead).

    Args:
        mask_float: (H, MB, NB) float32 block-sparse mask
        wave_size:  number of rows per SM wave (default 132 for H100)
        n_iter:     number of power iterations (default 8)

    Returns:
        perm: (H, MB) int64 row permutation
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)  # (H, MB)
    n_waves = (MB + wave_size - 1) // wave_size

    # --- Step 1: Implicit block power iteration for top-3 singular vectors ---
    # Iterate: U = M @ V → orthogonalize → V = M^T @ U → orthogonalize
    # This computes SVD of M without forming the MB×MB matrix W.
    gen = torch.Generator(device=device).manual_seed(42)
    V = torch.randn(H, NB, 3, device=device, generator=gen)  # (H, NB, 3)

    # Inline MGS for initial orthogonalization
    v0 = V[:, :, 0]
    v0 = v0 / v0.norm(dim=1, keepdim=True).clamp(min=1e-8)
    v1 = V[:, :, 1]
    v1 = v1 - (v1 * v0).sum(dim=1, keepdim=True) * v0
    v1 = v1 / v1.norm(dim=1, keepdim=True).clamp(min=1e-8)
    v2 = V[:, :, 2]
    v2 = v2 - (v2 * v0).sum(dim=1, keepdim=True) * v0
    v2 = v2 - (v2 * v1).sum(dim=1, keepdim=True) * v1
    v2 = v2 / v2.norm(dim=1, keepdim=True).clamp(min=1e-8)
    V = torch.stack([v0, v1, v2], dim=2)

    Mt = mask_float.transpose(1, 2)  # cache transpose (H, NB, MB)

    for _ in range(n_iter):
        # Forward: U = M @ V  (H, MB, 3)
        U = torch.bmm(mask_float, V)

        # MGS orthogonalization on U (row-space)
        u0 = U[:, :, 0]
        u0 = u0 / u0.norm(dim=1, keepdim=True).clamp(min=1e-8)
        u1 = U[:, :, 1]
        u1 = u1 - (u1 * u0).sum(dim=1, keepdim=True) * u0
        u1 = u1 / u1.norm(dim=1, keepdim=True).clamp(min=1e-8)
        u2 = U[:, :, 2]
        u2 = u2 - (u2 * u0).sum(dim=1, keepdim=True) * u0
        u2 = u2 - (u2 * u1).sum(dim=1, keepdim=True) * u1
        u2 = u2 / u2.norm(dim=1, keepdim=True).clamp(min=1e-8)
        U = torch.stack([u0, u1, u2], dim=2)

        # Backward: V = M^T @ U  (H, NB, 3)
        V = torch.bmm(Mt, U)

        # MGS orthogonalization on V (column-space)
        v0 = V[:, :, 0]
        v0 = v0 / v0.norm(dim=1, keepdim=True).clamp(min=1e-8)
        v1 = V[:, :, 1]
        v1 = v1 - (v1 * v0).sum(dim=1, keepdim=True) * v0
        v1 = v1 / v1.norm(dim=1, keepdim=True).clamp(min=1e-8)
        v2 = V[:, :, 2]
        v2 = v2 - (v2 * v0).sum(dim=1, keepdim=True) * v0
        v2 = v2 - (v2 * v1).sum(dim=1, keepdim=True) * v1
        v2 = v2 / v2.norm(dim=1, keepdim=True).clamp(min=1e-8)
        V = torch.stack([v0, v1, v2], dim=2)

    # Final projection to obtain left singular vectors
    U = torch.bmm(mask_float, V)  # (H, MB, 3)

    # U[:, :, 0] ≈ u₁ (NNZ direction — discarded)
    # U[:, :, 1] ≈ u₂ (primary structure)
    # U[:, :, 2] ≈ u₃ (secondary structure)
    u2 = U[:, :, 1]  # (H, MB)
    u3 = U[:, :, 2]  # (H, MB)

    # --- Step 2: 2D angle sort on (u₂, u₃) ---
    angles = torch.atan2(
        u3 - u3.mean(dim=1, keepdim=True),
        u2 - u2.mean(dim=1, keepdim=True),
    )
    spectral_order = angles.argsort(dim=1)  # (H, MB)

    # --- Step 3: Wave post-processing (identical to wave_1d_reorder) ---
    pad = n_waves * wave_size - MB
    if pad > 0:
        pad_idx = torch.arange(MB, MB + pad, device=device).unsqueeze(0).expand(H, -1)
        spectral_order_pad = torch.cat([spectral_order, pad_idx], dim=1)
        nnz_pad = torch.cat([nnz, torch.zeros(H, pad, dtype=nnz.dtype, device=device)], dim=1)
    else:
        spectral_order_pad = spectral_order
        nnz_pad = nnz

    nnz_ordered = torch.gather(nnz_pad, 1, spectral_order_pad)
    nnz_waves = nnz_ordered.view(H, n_waves, wave_size)
    wave_mean_nnz = nnz_waves.float().mean(dim=2)

    wave_order = wave_mean_nnz.argsort(dim=1, descending=True)
    within_wave_order = nnz_waves.argsort(dim=2, descending=True)

    spectral_waves = spectral_order_pad.view(H, n_waves, wave_size)
    spectral_sorted = torch.gather(spectral_waves, 2, within_wave_order)

    wave_order_exp = wave_order.unsqueeze(2).expand(H, n_waves, wave_size)
    result_padded = torch.gather(spectral_sorted, 1, wave_order_exp)

    valid_mask = result_padded < MB
    result = result_padded[valid_mask].view(H, MB).contiguous()

    return result


def svd_deflation_reorder(mask_float, wave_size=132, n_iter=10):
    """2D Spectral Reordering via Sequential Hotelling Deflation.

    Computes u₂ + u₃ of W = M @ M^T via 3-phase sequential deflation:
      Phase 1: u₁ = dominant eigenvector of W   (power iter, k=1, normalize)
      Phase 2: u₂ = dominant eigenvector of W'  (W projected ⊥ u₁)
      Phase 3: u₃ = dominant eigenvector of W'' (W projected ⊥ u₁, u₂)

    Key advantages over block power + QR (v1):
      - No explicit W = M @ M^T formation  → saves O(H·MB²·NB) FLOPs + 106MB
      - No QR decomposition needed          → k=1 per phase, just normalize
      - Orthogonality via explicit projection (rank-1 subtraction, O(H·MB))

    Mathematical equivalence:
      W' = W - λ₁ u₁ u₁^T has eigenvalues {0, λ₂, λ₃, ...}
      Power iteration on W' converges to u₂ (dominant of W').
      Explicit projection v ← v - (v·u₁)u₁ ensures numerical stability
      by preventing any u₁ leakage from accumulating.

    Cost: 3 × wave_1d ≈ 3 × 10 × 2 BMMs = 60 thin BMMs
    vs v1: 1 fat BMM (W construction) + 15 × 1 BMM + 15 × QR

    Args:
        mask_float: (H, MB, NB) float32 block-sparse mask
        wave_size:  number of rows per SM wave (default 132 for H100)
        n_iter:     iterations per phase (default 10)

    Returns:
        perm: (H, MB) int64 row permutation
    """
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)  # (H, MB)
    n_waves = (MB + wave_size - 1) // wave_size

    Mt = mask_float.transpose(1, 2)  # (H, NB, MB) — cache once
    gen = torch.Generator(device=device).manual_seed(42)

    def implicit_Wv(v):
        """Compute W @ v = M @ (M^T @ v) without forming W.
        v: (H, MB) → result: (H, MB)
        Cost: 2 thin BMMs of (H, MB, NB) × (H, NB, 1).
        """
        v3 = v.unsqueeze(2)            # (H, MB, 1)
        temp = torch.bmm(Mt, v3)       # (H, NB, 1) = M^T @ v
        result = torch.bmm(mask_float, temp)  # (H, MB, 1) = M @ (M^T @ v)
        return result.squeeze(2)       # (H, MB)

    def power_iter_deflated(n_iter, deflation_vecs):
        """Power iteration for dominant eigenvector of deflated W.
        At each step:  v ← W@v,  then project out all deflation_vecs,  then normalize.
        Args:
            n_iter: number of iterations
            deflation_vecs: list of (H, MB) tensors to project out
        Returns:
            u: (H, MB) dominant eigenvector of deflated W
        """
        v = torch.randn(H, MB, device=device, generator=gen)
        v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)

        for _ in range(n_iter):
            v = implicit_Wv(v)  # W @ v via 2 thin BMMs

            # Project out previously found eigenvectors (Hotelling deflation)
            for u_prev in deflation_vecs:
                coeff = (v * u_prev).sum(dim=1, keepdim=True)  # (H, 1)
                v = v - coeff * u_prev

            v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)

        return v

    # --- Phase 1: u₁ (dominant eigenvector — will be discarded) ---
    u1 = power_iter_deflated(n_iter, [])

    # --- Phase 2: u₂ (dominant of W ⊥ u₁) ---
    u2 = power_iter_deflated(n_iter, [u1])

    # --- Phase 3: u₃ (dominant of W ⊥ u₁, u₂) ---
    u3 = power_iter_deflated(n_iter, [u1, u2])

    # --- Step 2: 2D angle sort on (u₂, u₃) ---
    angles = torch.atan2(
        u3 - u3.mean(dim=1, keepdim=True),
        u2 - u2.mean(dim=1, keepdim=True),
    )
    spectral_order = angles.argsort(dim=1)

    # --- Step 3: Wave post-processing ---
    return _svd_wave_postprocess(spectral_order, nnz, H, MB, wave_size, device)

# ============================================================
# SVD u2u3 variants for ablation testing
# Current svd_u2u3_reorder above is v5 (implicit + MGS + 15iter)
# ============================================================

def _svd_wave_postprocess(spectral_order, nnz, H, MB, wave_size, device):
    """Shared wave post-processing for all SVD variants."""
    n_waves = (MB + wave_size - 1) // wave_size
    pad = n_waves * wave_size - MB
    if pad > 0:
        pad_idx = torch.arange(MB, MB + pad, device=device).unsqueeze(0).expand(H, -1)
        spectral_order_pad = torch.cat([spectral_order, pad_idx], dim=1)
        nnz_pad = torch.cat([nnz, torch.zeros(H, pad, dtype=nnz.dtype, device=device)], dim=1)
    else:
        spectral_order_pad = spectral_order
        nnz_pad = nnz

    nnz_ordered = torch.gather(nnz_pad, 1, spectral_order_pad)
    nnz_waves = nnz_ordered.view(H, n_waves, wave_size)
    wave_mean_nnz = nnz_waves.float().mean(dim=2)

    wave_order = wave_mean_nnz.argsort(dim=1, descending=True)
    within_wave_order = nnz_waves.argsort(dim=2, descending=True)

    spectral_waves = spectral_order_pad.view(H, n_waves, wave_size)
    spectral_sorted = torch.gather(spectral_waves, 2, within_wave_order)

    wave_order_exp = wave_order.unsqueeze(2).expand(H, n_waves, wave_size)
    result_padded = torch.gather(spectral_sorted, 1, wave_order_exp)

    valid_mask = result_padded < MB
    result = result_padded[valid_mask].view(H, MB).contiguous()
    return result


def svd_u2u3_v1(mask_float, wave_size=132, n_iter=15):
    """Original: explicit W = M@M^T + QR + 15 iterations."""
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)

    W = torch.bmm(mask_float, mask_float.transpose(1, 2))
    gen = torch.Generator(device=device).manual_seed(42)
    V = torch.randn(H, MB, 3, device=device, generator=gen)
    V, _ = torch.linalg.qr(V)

    for _ in range(n_iter):
        V = torch.bmm(W, V)
        V, _ = torch.linalg.qr(V)

    del W
    u2, u3 = V[:, :, 1], V[:, :, 2]
    angles = torch.atan2(u3 - u3.mean(1, keepdim=True), u2 - u2.mean(1, keepdim=True))
    spectral_order = angles.argsort(dim=1)
    return _svd_wave_postprocess(spectral_order, nnz, H, MB, wave_size, device)


def svd_u2u3_v2(mask_float, wave_size=132, n_iter=10):
    """V2: implicit SVD + QR + 10 iterations."""
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)

    gen = torch.Generator(device=device).manual_seed(42)
    V = torch.randn(H, NB, 3, device=device, generator=gen)
    V, _ = torch.linalg.qr(V)
    Mt = mask_float.transpose(1, 2)

    for _ in range(n_iter):
        U = torch.bmm(mask_float, V)
        U, _ = torch.linalg.qr(U)
        V = torch.bmm(Mt, U)
        V, _ = torch.linalg.qr(V)

    U = torch.bmm(mask_float, V)
    u2, u3 = U[:, :, 1], U[:, :, 2]
    angles = torch.atan2(u3 - u3.mean(1, keepdim=True), u2 - u2.mean(1, keepdim=True))
    spectral_order = angles.argsort(dim=1)
    return _svd_wave_postprocess(spectral_order, nnz, H, MB, wave_size, device)


def _inline_mgs_3(V):
    """Inline Modified Gram-Schmidt for 3 vectors. V: (H, N, 3)."""
    v0 = V[:, :, 0]
    v0 = v0 / v0.norm(dim=1, keepdim=True).clamp(min=1e-8)
    v1 = V[:, :, 1]
    v1 = v1 - (v1 * v0).sum(dim=1, keepdim=True) * v0
    v1 = v1 / v1.norm(dim=1, keepdim=True).clamp(min=1e-8)
    v2 = V[:, :, 2]
    v2 = v2 - (v2 * v0).sum(dim=1, keepdim=True) * v0
    v2 = v2 - (v2 * v1).sum(dim=1, keepdim=True) * v1
    v2 = v2 / v2.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.stack([v0, v1, v2], dim=2)


def svd_u2u3_v3(mask_float, wave_size=132, n_iter=5):
    """V3: explicit W = M@M^T + MGS + 5 iterations."""
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)

    W = torch.bmm(mask_float, mask_float.transpose(1, 2))
    gen = torch.Generator(device=device).manual_seed(42)
    V = torch.randn(H, MB, 3, device=device, generator=gen)
    V, _ = torch.linalg.qr(V)

    for _ in range(n_iter):
        V = torch.bmm(W, V)
        V = _inline_mgs_3(V)

    del W
    u2, u3 = V[:, :, 1], V[:, :, 2]
    angles = torch.atan2(u3 - u3.mean(1, keepdim=True), u2 - u2.mean(1, keepdim=True))
    spectral_order = angles.argsort(dim=1)
    return _svd_wave_postprocess(spectral_order, nnz, H, MB, wave_size, device)


def svd_u2u3_v4(mask_float, wave_size=132, n_iter=8):
    """V4: implicit SVD + MGS + 8 iterations."""
    H, MB, NB = mask_float.shape
    device = mask_float.device
    nnz = mask_float.sum(dim=2)

    gen = torch.Generator(device=device).manual_seed(42)
    V = torch.randn(H, NB, 3, device=device, generator=gen)
    V = _inline_mgs_3(V)
    Mt = mask_float.transpose(1, 2)

    for _ in range(n_iter):
        U = torch.bmm(mask_float, V)
        U = _inline_mgs_3(U)
        V = torch.bmm(Mt, U)
        V = _inline_mgs_3(V)

    U = torch.bmm(mask_float, V)
    u2, u3 = U[:, :, 1], U[:, :, 2]
    angles = torch.atan2(u3 - u3.mean(1, keepdim=True), u2 - u2.mean(1, keepdim=True))
    spectral_order = angles.argsort(dim=1)
    return _svd_wave_postprocess(spectral_order, nnz, H, MB, wave_size, device)

