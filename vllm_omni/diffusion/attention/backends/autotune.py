"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
import numpy as np
from spas_sage_attn.utils import precision_metric
from spas_sage_attn import spas_sage_attn_meansim_cuda, spas_sage2_attn_meansim_cuda
import warnings
from einops import rearrange

def extract_sparse_attention_state_dict(model, verbose=False):
    saved_state_dict = {}
    for k, v in model.named_modules(): # enumerate all nn.Module instance in the model
        if isinstance(v, SparseAttentionMeansim):
            if verbose: print(k, 'is an instance of SparseAttentionMeansim')
            for model_key, model_param in model.state_dict().items(): # find the corresponding state_dict item
                if k in model_key:
                    if verbose: print(f'{model_key} is a substate_dict of {k}, we will save it.')
                    saved_state_dict[model_key] = model_param
    return saved_state_dict


def load_sparse_attention_state_dict(model, saved_state_dict, multigpu=False, verbose=False):
    if not multigpu:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    for k, v in model.named_modules():
        if isinstance(v, SparseAttentionMeansim): # find each SparseAttentionMeansim instance
            if verbose: print(k, 'is an instance of SparseAttentionMeansim, but it is empty now.')
            for sk, sv in saved_state_dict.items():
                if k in sk:
                    if verbose: print(f'{sk} is a substate_dict of {k}, we will load it.')
                    sub_name = sk.split(k)[1][1:]
                    if multigpu:
                        sv= sv.to(device=v.device)
                    else:
                        sv = sv.to(device=device, dtype=dtype)
                    setattr(v, sub_name, nn.Parameter(sv, requires_grad=False))
    if not multigpu:
        model = model.to(device)
    return model


def partition_points_into_line(points, block_size, min_dim1=-1, max_dim1=1):
    blocks = {}
    for point in points:
        dim1 = point['simthreshd1']
        # Calculate block indices for dim1 and dim2
        block_index_dim1 = int((dim1 - min_dim1) // block_size)
        key = (block_index_dim1,)
        # Initialize the block if it doesn't exist
        if key not in blocks:
            blocks[key] = []
        blocks[key].append(point)
    return blocks

# 
from tools.gpu_process import GPUProcessPoolExecutor
executor = GPUProcessPoolExecutor()

class SparseAttentionMeansim(nn.Module):
    def __init__(self, sim_rule="l1", l1=0.07, pv_l1=0.08, cos_sim=0.98, rmse=0.07, rearrange_kwargs={}, tune_pv=False):
        super(SparseAttentionMeansim, self).__init__()
        self.head_num = None
        assert l1 >= 0 and cos_sim <= 1 and rmse >= 0, "l1, cos_sim, rmse should be legal"
        assert pv_l1 > l1, 'pv l1 must greater than l1'
        self.l1 = l1
        self.pv_l1 = pv_l1
        self.cos_sim = cos_sim
        self.rmse = rmse
        self.is_sparse = None  # bool, shape of head number, decide whether to use sparse attention for each head
        self.cdfthreshd = None  # float, shape of head number, decide the threshold of cdf for each head
        self.simthreshd1 = None
        self.simthreshd2 = None
        self.pvthreshd = None
        self.tuning_sparsity = None
        self.num_data_passed = 0
        self.hyperparams_cache = {}
        self.sim_rule = sim_rule
        self.rearrange_kwargs = rearrange_kwargs
        self.tune_pv = tune_pv

    @staticmethod
    def _is_rank0() -> bool:
        return os.environ.get("RANK", "0") == "0"

    @staticmethod
    def _log_level() -> int:
        # 0: off, 1: head/phase, 2: per-sim candidate, 3: per binary-search iteration
        try:
            return int(os.environ.get("SPARGE_AUTOTUNE_LOG_LEVEL", "2"))
        except Exception:
            return 2

    def _debug_log(self, level: int, message: str) -> None:
        if not self._is_rank0():
            return
        if self._log_level() >= level:
            print(f"[autotune-debug:L{level}] {message}", flush=True)

    @staticmethod
    def _time_log_level() -> int:
        # 0: off, 1: per-head summary, 2: per-candidate summary, 3: per-iteration detail
        try:
            return int(os.environ.get("SPARGE_AUTOTUNE_TIME_LOG_LEVEL", "0"))
        except Exception:
            return 0

    def _time_log(self, level: int, message: str) -> None:
        if not self._is_rank0():
            return
        if self._time_log_level() >= level:
            print(f"[autotune-time:L{level}] {message}", flush=True)

    @staticmethod
    def _sim_granularity() -> int:
        # Number of sim bins per side in [-1, 1). Total candidates = 2 * granularity.
        # Default 8 halves search points vs previous default 16.
        try:
            granularity = int(os.environ.get("SPARGE_AUTOTUNE_SIM_GRANULARITY", "4"))
        except Exception:
            granularity = 4
        return max(1, granularity)

    @staticmethod
    def _metric_value(metrics: dict, key: str) -> float:
        value = metrics[key]
        if isinstance(value, torch.Tensor):
            value = value.item()
        return float(value)

    def _compute_metrics(self, o_gt, o_sparse) -> dict:
        metrics = precision_metric(o_sparse, o_gt, verbose=False)
        return {
            "L1": self._metric_value(metrics, "L1"),
            "RMSE": self._metric_value(metrics, "RMSE"),
            "Cossim": self._metric_value(metrics, "Cossim"),
        }

    def _is_sim_from_metrics(self, metrics: dict) -> bool:
        if self.sim_rule == "cosine":
            return metrics["Cossim"] > self.cos_sim
        elif self.sim_rule == "rmse":
            return metrics["RMSE"] < self.rmse
        elif self.sim_rule == "l1":
            return metrics["L1"] < self.l1
        else:
            raise ValueError("sim_rule should be one of ['cosine', 'rmse', 'l1']")

    def _format_metrics(self, metrics: dict) -> str:
        return (
            f"L1={metrics['L1']:.6f} RMSE={metrics['RMSE']:.6f} "
            f"Cossim={metrics['Cossim']:.6f}"
        )

    @staticmethod
    def _format_point(point: dict) -> str:
        return (
            "{"
            f"sim={point['simthreshd1']:.4f}, "
            f"cdf={point['cdfthreshd']:.6f}, "
            f"pv={point['pvthreshd']:.6f}, "
            f"sparsity={float(point['sparsity']):.6f}, "
            f"data_idx={point['data_idx']}"
            "}"
        )

    def _log_group_selection(self, head_idx: int, groups: list, final_group: list) -> None:
        if self._log_level() < 1 or not self._is_rank0():
            return
        topk = 3
        self._debug_log(
            1,
            f"head={head_idx} group-summary total_groups={len(groups)} final_group_size={len(final_group)}",
        )
        for rank, group in enumerate(groups[:topk], start=1):
            sum_sparsity = float(sum(float(point['sparsity']) for point in group))
            mean_sparsity = float(np.mean([float(point['sparsity']) for point in group]))
            sim_min = min(float(point['simthreshd1']) for point in group)
            sim_max = max(float(point['simthreshd1']) for point in group)
            label = "winner" if group is final_group else "runner-up"
            points = ", ".join(self._format_point(point) for point in group)
            self._debug_log(
                1,
                f"head={head_idx} group-rank={rank} label={label} size={len(group)} "
                f"sum_sparsity={sum_sparsity:.6f} mean_sparsity={mean_sparsity:.6f} "
                f"sim_range=({sim_min:.4f},{sim_max:.4f}) points=[{points}]",
            )
    
    def is_sim(self, o_gt, o_sparse):
        metrics = self._compute_metrics(o_gt, o_sparse)
        return self._is_sim_from_metrics(metrics)
    
    def init_hyperparams(self, head_num, device):
        self.head_num = head_num
        self.is_sparse = nn.Parameter(
            torch.ones(self.head_num, dtype=torch.bool, device=device),
            requires_grad=False,
        )
        self.cdfthreshd = nn.Parameter(
            torch.ones(self.head_num, device=device) * 0.1,
            requires_grad=False,
        )
        self.simthreshd1 = nn.Parameter(
            torch.ones(self.head_num, device=device) * -1,
            requires_grad=False,
        )
        self.simthreshd2 = nn.Parameter(
            torch.zeros(self.head_num, device=device),
            requires_grad=False,
        )
        self.pvthreshd = nn.Parameter(
            torch.ones(self.head_num, device=device) * 20,
            requires_grad=False,
        )
        self.tuning_sparsity = torch.zeros(self.head_num, device=device)
        self.num_data_passed = 0
        self.hyperparams_cache = {}

    def kernel_selection(self):
        sm = torch.cuda.get_device_capability()
        sm = 10*sm[0] + sm[1]
        if sm >= 89:
            return spas_sage2_attn_meansim_cuda
        else:
            warnings.warn(f'{sm=}, do not support sageattn2, using sageattn1 kernel')
            return spas_sage_attn_meansim_cuda

    @torch.no_grad()
    def tune_pvthreshd(self, qi, ki, vi, mask=None, is_causal=False, smooth_k=True,
                      simthreshd1=None, cdfthreshd=None, head_idx=None):
        gt_i = F.scaled_dot_product_attention(qi, ki, vi, mask, is_causal=is_causal)
        cur_min_pvthreshd = 0.0
        cur_max_pvthreshd = 50.0
        cur_pvthreshd = 20.0
        delta = 0.1
        iter_idx = 0
        metrics = None
        self._debug_log(
            2,
            f"head={head_idx} pv-search start sim={simthreshd1:.4f} cdf={cdfthreshd:.6f} "
            f"range=({cur_min_pvthreshd:.3f},{cur_max_pvthreshd:.3f})",
        )
        while cur_max_pvthreshd - cur_min_pvthreshd > delta:
            iter_idx += 1
            self._debug_log(
                3,
                f"head={head_idx} pv iter={iter_idx} before-kernel pv={cur_pvthreshd:.6f} "
                f"range=({cur_min_pvthreshd:.6f},{cur_max_pvthreshd:.6f})",
            )
            kernel = self.kernel_selection()
            sparse_i, sparsity = kernel(
                qi,
                ki,
                vi,
                mask,
                is_causal=is_causal,
                cdfthreshd=cdfthreshd,
                smooth_k=smooth_k,
                return_sparsity=True,
                simthreshd1=simthreshd1 if simthreshd1 is not None else self.simthreshd1,
                pvthreshd=cur_pvthreshd,
            )
            metrics = self._compute_metrics(gt_i, sparse_i)
            self._debug_log(
                3,
                f"head={head_idx} pv iter={iter_idx} after-kernel pv={cur_pvthreshd:.6f} "
                f"{self._format_metrics(metrics)} sparsity={float(sparsity):.6f}",
            )
            if metrics["L1"] < self.pv_l1:
                cur_max_pvthreshd = cur_pvthreshd
                cur_pvthreshd = (cur_pvthreshd + cur_min_pvthreshd) / 2
            else:
                cur_min_pvthreshd = cur_pvthreshd
                cur_pvthreshd = (cur_pvthreshd + cur_max_pvthreshd) / 2
        self._debug_log(
            2,
            f"head={head_idx} pv-search done sim={simthreshd1:.4f} pv={cur_pvthreshd:.6f} "
            f"iters={iter_idx} {self._format_metrics(metrics)} final_sparsity={float(sparsity):.6f}",
        )
        return cur_pvthreshd, sparsity

    @torch.no_grad()
    def tune_cdfthreshd(
        self, qi, ki, vi, mask=None, is_causal=False, smooth_k=True, simthreshd1=None, head_idx=None
    ):
        t_cdf_start = time.perf_counter()
        t_kernel_total = 0.0
        t_metric_total = 0.0
        gt_i = F.scaled_dot_product_attention(qi, ki, vi, mask, is_causal=is_causal)
        cur_min_cdfthreshd = 0.0
        cur_max_cdfthreshd = 1.0
        cur_cdfthreshd = 0.1
        try:
            delta = float(os.environ.get("SPARGE_AUTOTUNE_CDF_DELTA", "0.001"))
        except Exception:
            delta = 0.001
        iter_idx = 0
        metrics = None
        self._debug_log(
            2,
            f"head={head_idx} cdf-search start sim={simthreshd1:.4f} "
            f"range=({cur_min_cdfthreshd:.3f},{cur_max_cdfthreshd:.3f})",
        )
        import inspect
        # 获取FlashAttentionImpl.layeridx
        try:
            from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
            layeridx = getattr(FlashAttentionImpl, "layeridx", 0)
        except Exception:
            layeridx = 0
        rank = os.environ.get("RANK", "?")
        enable_loop_log = False
        # 受控于环境变量 SPARGE_AUTOTUNE_CDF_LOOP_LOG 和 layeridx>30
        if os.environ.get("SPARGE_AUTOTUNE_CDF_LOOP_LOG", "0") == "1" and layeridx > 30:
            enable_loop_log = True
        while cur_max_cdfthreshd - cur_min_cdfthreshd > delta:
            iter_idx += 1
            self._debug_log(
                3,
                f"head={head_idx} cdf iter={iter_idx} before-kernel cdf={cur_cdfthreshd:.6f} "
                f"range=({cur_min_cdfthreshd:.6f},{cur_max_cdfthreshd:.6f})",
            )
            # kernel = spas_sage2_attn_meansim_cuda
            # t_kernel_start = time.perf_counter()
            sparse_i, sparsity = spas_sage2_attn_meansim_cuda(
                qi,
                ki,
                vi,
                mask,
                is_causal=is_causal,
                cdfthreshd=cur_cdfthreshd,
                smooth_k=smooth_k,
                return_sparsity=True,
                simthreshd1=simthreshd1 if simthreshd1 is not None else self.simthreshd1,
            )
            # t_kernel_total += (time.perf_counter() - t_kernel_start)
            t_kernel_total = 0.0
            t_metric_start = time.perf_counter()
            metrics = self._compute_metrics(gt_i, sparse_i)
            t_metric_total += (time.perf_counter() - t_metric_start)
            sim_ok = self._is_sim_from_metrics(metrics)
            self._debug_log(
                3,
                f"head={head_idx} cdf iter={iter_idx} after-kernel cdf={cur_cdfthreshd:.6f} "
                f"sim_ok={int(sim_ok)} {self._format_metrics(metrics)} "
                f"thresholds=(l1<{self.l1:.6f}, rmse<{self.rmse:.6f}, cos>{self.cos_sim:.6f}) "
                f"sparsity={float(sparsity):.6f}",
            )
            # if enable_loop_log:
            #     print(f"[LOOP_LOG][RANK={rank}] layeridx={layeridx} head={head_idx} iter={iter_idx} cdf={cur_cdfthreshd:.6f} min={cur_min_cdfthreshd:.6f} max={cur_max_cdfthreshd:.6f} sim_ok={int(sim_ok)} sparsity={float(sparsity):.6f} metrics={self._format_metrics(metrics)}", flush=True)
            if self._time_log_level() >= 3:
                self._time_log(
                    3,
                    f"head={head_idx} sim={simthreshd1:.4f} cdf_iter={iter_idx} "
                    f"kernel_ms={(t_kernel_total * 1000.0):.2f} "
                    f"metric_ms={(t_metric_total * 1000.0):.2f}",
                )
            if sim_ok:
                cur_max_cdfthreshd = cur_cdfthreshd
                cur_cdfthreshd = (cur_cdfthreshd + cur_min_cdfthreshd) / 2
            else:
                cur_min_cdfthreshd = cur_cdfthreshd
                cur_cdfthreshd = (cur_cdfthreshd + cur_max_cdfthreshd) / 2

        if cur_cdfthreshd > 1 - delta:  # could not reach precision, using full attention
            cur_cdfthreshd = 1
        elif cur_cdfthreshd < delta:
            # no sim block is already enough for precision, mostly not sparse, using full attention
            # suggest to use more data to tune or use full attention
            pass
        self._debug_log(
            2,
            f"head={head_idx} cdf-search done sim={simthreshd1:.4f} cdf={cur_cdfthreshd:.6f} "
            f"iters={iter_idx} {self._format_metrics(metrics)} final_sparsity={float(sparsity):.6f}",
        )
        t_cdf_total = time.perf_counter() - t_cdf_start
        t_other = max(0.0, t_cdf_total - t_kernel_total - t_metric_total)
        if self._time_log_level() >= 2:
            self._time_log(
                2,
                f"head={head_idx} sim={simthreshd1:.4f} cdf_done iters={iter_idx} "
                f"total_ms={(t_cdf_total * 1000.0):.2f} "
                f"kernel_ms={(t_kernel_total * 1000.0):.2f} "
                f"metric_ms={(t_metric_total * 1000.0):.2f} "
                f"other_ms={(t_other * 1000.0):.2f} "
                f"avg_iter_ms={((t_cdf_total / max(iter_idx, 1)) * 1000.0):.2f}",
            )
        # if enable_loop_log:
        #     print(f"[autotune][RANK={rank}] [tune_cdfthreshold] layeridx={layeridx} head={head_idx if 'head_idx' in locals() else ''} END", flush=True)
        return cur_cdfthreshd, sparsity

    @torch.no_grad()
    def autotune(self, qi, ki, vi, head_idx, mask=None, is_causal=False, smooth_k=True):
        t_head_start = time.perf_counter()
        qi = qi.to(torch.cuda.current_device())
        ki = ki.to(torch.cuda.current_device())
        vi = vi.to(torch.cuda.current_device())
        all_hyperparams = []
        granularity = self._sim_granularity()
        cdf_time_acc = 0.0
        cand_count = 0
        self._debug_log(
            1,
            f"head={head_idx} autotune start q={tuple(qi.shape)} k={tuple(ki.shape)} v={tuple(vi.shape)} "
            f"tune_pv={int(self.tune_pv)} sim_granularity={granularity} sim_candidates={2 * granularity}",
        )
        rank = os.environ.get("RANK", "?")
        for simthreshd1 in range(int(-1 * granularity), int(1 * granularity)):
            t_cand_start = time.perf_counter()
            simthreshd1 = simthreshd1 / granularity
            self._debug_log(2, f"head={head_idx} sim-candidate start sim={simthreshd1:.4f}")
            # print(f"[autotune][RANK={rank}] head={head_idx} sim-candidate start sim={simthreshd1:.4f}", flush=True)
            t_cdf_call_start = time.perf_counter()
            cur_cdfthreshd, sparsity = self.tune_cdfthreshd(
                qi,
                ki,
                vi,
                mask,
                is_causal=is_causal,
                smooth_k=smooth_k,
                simthreshd1=simthreshd1,
                head_idx=head_idx,
            )
            # print(f"[autotune][RANK={rank}] head={head_idx} sim-candidate cdf done sim={simthreshd1:.4f}", flush=True)
            cdf_elapsed = time.perf_counter() - t_cdf_call_start
            cdf_time_acc += cdf_elapsed
            if self.tune_pv:
                pvthreshd, _ = self.tune_pvthreshd(
                    qi,
                    ki,
                    vi,
                    mask,
                    is_causal=is_causal,
                    smooth_k=smooth_k,
                    simthreshd1=simthreshd1,
                    cdfthreshd=cur_cdfthreshd,
                    head_idx=head_idx,
                )
            else:
                pvthreshd = 20
            all_hyperparams.append({
                "simthreshd1": simthreshd1,
                "cdfthreshd": cur_cdfthreshd,
                'pvthreshd': pvthreshd,
                "sparsity": sparsity,
                'data_idx': self.num_data_passed
            })
            self._debug_log(
                2,
                f"head={head_idx} sim-candidate done sim={simthreshd1:.4f} "
                f"cdf={cur_cdfthreshd:.6f} pv={pvthreshd:.6f} sparsity={float(sparsity):.6f}",
            )
            cand_count += 1
            if self._time_log_level() >= 2:
                t_cand_total = time.perf_counter() - t_cand_start
                self._time_log(
                    2,
                    f"head={head_idx} sim={simthreshd1:.4f} candidate_done "
                    f"cand_ms={(t_cand_total * 1000.0):.2f} cdf_ms={(cdf_elapsed * 1000.0):.2f} "
                    f"cdf_ratio={(cdf_elapsed / max(t_cand_total, 1e-9)):.3f} sparsity={float(sparsity):.6f}",
                )
            if sparsity < 0.1:
                self._debug_log(
                    2,
                    f"head={head_idx} early-stop sim={simthreshd1:.4f} sparsity={float(sparsity):.6f}",
                )
                break  # no need to continue to raise threshold bound
        if self.hyperparams_cache.get(head_idx) is None:
            self.hyperparams_cache[head_idx] = []
        cache_hyper = self.hyperparams_cache[head_idx]
        all_hyperparams = all_hyperparams + cache_hyper
        self.hyperparams_cache[head_idx] = all_hyperparams

        grid = partition_points_into_line(all_hyperparams, 2/granularity)
        groups = list(grid.values())
        # sort by sum of sparsity, local smoothing
        groups = sorted(groups, key=lambda x: sum([y['sparsity'] for y in x]), reverse=True)
        final_group = groups[0]
        self._log_group_selection(head_idx, groups, final_group)
        final_simthreshd1 = np.max([x['simthreshd1'] for x in final_group]).item()
        final_cdfthreshd = np.max([x['cdfthreshd'] for x in final_group]).item()
        final_pvthreshd = np.max([x['pvthreshd'] for x in final_group]).item()
        mean_sparsity = np.mean([x['sparsity'] for x in final_group]).item()
        self._debug_log(
            1,
            f"head={head_idx} autotune done final(sim={final_simthreshd1:.4f}, "
            f"cdf={final_cdfthreshd:.6f}, pv={final_pvthreshd:.6f}, sparsity={mean_sparsity:.6f})",
        )
        if self._time_log_level() >= 1:
            t_head_total = time.perf_counter() - t_head_start
            self._time_log(
                1,
                f"head={head_idx} head_done total_ms={(t_head_total * 1000.0):.2f} "
                f"cdf_total_ms={(cdf_time_acc * 1000.0):.2f} "
                f"cdf_ratio={(cdf_time_acc / max(t_head_total, 1e-9)):.3f} "
                f"candidates={cand_count} avg_cand_ms={((t_head_total / max(cand_count, 1)) * 1000.0):.2f}",
            )
        return {
            'final_simthreshd1': final_simthreshd1,
            'final_cdfthreshd': final_cdfthreshd,
            'final_pvthreshd': final_pvthreshd,
            'mean_sparsity': mean_sparsity,
            'head_idx': head_idx
        }
        
    def fill_results(self, rtdict):
        head_idx = rtdict['head_idx']
        self.simthreshd1[head_idx] = rtdict['final_simthreshd1']
        self.cdfthreshd[head_idx] = rtdict['final_cdfthreshd']
        self.pvthreshd[head_idx] = rtdict['final_pvthreshd']
        self.is_sparse[head_idx] = rtdict['mean_sparsity'] > 0.1 and self.is_sparse[head_idx]
        self.tuning_sparsity[head_idx] = rtdict['mean_sparsity']
        if not self.is_sparse[head_idx]:
            self.cdfthreshd[head_idx] = 1
            self.simthreshd1[head_idx] = 1
        
    @torch.no_grad()
    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        is_causal=False,
        scale=None,
        tensor_layout="HND",
        tune_mode=False,
        smooth_k=True,
        return_sparsity=False,
    ):
        assert len(q.shape) == 4, "q should be 4-d tensor with B, H, L, D"
            
        if os.environ.get("TUNE_MODE", "") != "" or tune_mode:
            if tensor_layout == 'NHD':
                q = rearrange(q, '... L H D -> ... H L D')
                k = rearrange(k, '... L H D -> ... H L D')
                v = rearrange(v, '... L H D -> ... H L D')
            if self.is_sparse is None:  # init per head hyper parameters
                self.init_hyperparams(q.shape[1], q.device)
            if os.environ.get('PARALLEL_TUNE', '') == '':
                for i in tqdm(range(self.head_num)):
                    if not self.is_sparse[i].item():
                        continue
                    qi, ki, vi = q[:, i : i + 1], k[:, i : i + 1], v[:, i : i + 1]
                    rtdict = self.autotune(qi, ki, vi, head_idx=i, mask=mask, is_causal=is_causal, smooth_k=smooth_k)
                    self.fill_results(rtdict)
            else:
                futures = []
                for i in range(self.head_num):
                    if not self.is_sparse[i].item():
                        continue
                    qi, ki, vi = q[:, i : i + 1], k[:, i : i + 1], v[:, i : i + 1]
                    future = executor.submit(self.autotune, qi, ki, vi, head_idx=i, mask=mask, is_causal=is_causal, smooth_k=smooth_k)
                    futures.append(future)
                for future in tqdm(futures):
                    rtdict = future.result()
                    self.fill_results(rtdict)
                    
                
            self.num_data_passed += 1
            print(f'{self.cdfthreshd=}')
            print(f'{self.simthreshd1=}')
            print(f'{self.is_sparse=}')
            print(f'{self.pvthreshd=}')
            print(f'{self.tuning_sparsity=}')
            print(f'mean sparsity:{self.tuning_sparsity.mean().item()}')
            o = F.scaled_dot_product_attention(q, k, v, mask, is_causal=is_causal)
            if tensor_layout == 'NHD':
                o = rearrange(o, '... H L D -> ... L H D')
            torch.cuda.empty_cache()
        else:
            assert self.cdfthreshd is not None, "attention hyperparameters should be tuned first"
            kernel = self.kernel_selection()
            o = kernel(
                q,
                k,
                v,
                mask,
                is_causal=is_causal,
                smooth_k=smooth_k,
                scale=scale,
                tensor_layout=tensor_layout,
                cdfthreshd=self.cdfthreshd,
                simthreshd1=self.simthreshd1,
                pvthreshd=self.pvthreshd.float(),
                return_sparsity=return_sparsity,
                attention_sink= True,  # Only keep True when inference !!!!
            )
        
        if return_sparsity:
            o, total_sparsity = o
            return o, total_sparsity
        else:
            return o
