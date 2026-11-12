"""Phase 2: NCU profiling with real dumped masks (BLK=128, TP=1).

Profiles FA4 kernel with no_reorder vs wave_1d on the highest-CV layers
per sparse_mode. Collects L2 hit rate, DRAM read, and kernel timing.

Usage:
    # Single profile (for NCU wrapping):
    python paper_ncu_profile.py --mask_dir /path/to/mask_dir --layer 38 --reorder wave_1d

    # Timing sweep across modes + layers:
    python paper_ncu_profile.py --sweep
"""
import torch
import sys
import os
import argparse
import json
import glob


MASK_ROOT = "/data1/dhelix_shared/daizijian/vllm-omni-mask-paper"


def load_mask(mask_dir, layer, threshold="0_9"):
    """Load a real BLK=128 mask from dumped files."""
    path = os.path.join(mask_dir, f"mask_{layer}_thr{threshold}", "mask_no_reorder.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")
    return torch.load(path, weights_only=False).cuda()


def get_reorder_perm(mask, mode):
    H, MB, NB = mask.shape
    if mode == "no_reorder":
        return torch.arange(MB, device='cuda').unsqueeze(0).expand(H, MB).contiguous()
    elif mode == "nnz_reorder":
        return torch.argsort(mask.float().sum(dim=2), descending=True, dim=1)
    elif mode == "wave_1d":
        from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import wave_overlap_reorder
        return wave_overlap_reorder(mask.float(), wave_size=132)
    elif mode == "svd_deflation":
        from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import svd_deflation_reorder
        return svd_deflation_reorder(mask.float(), wave_size=132)
    elif mode == "spectral_only":
        # Stage 1 only: sort by top-1 singular vector (no wave scheduling)
        mask_f = mask.float()
        device = mask_f.device
        gen = torch.Generator(device=device).manual_seed(42)
        V = torch.randn(H, NB, 1, device=device, generator=gen)
        for _ in range(10):
            U = torch.bmm(mask_f, V)
            U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-8)
            V = torch.bmm(mask_f.transpose(1, 2), U)
            V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return U[:, :, 0].argsort(dim=1, descending=True)
    elif mode == "wave_sched_only":
        # Stage 2 only: NNZ-based wave scheduling (no spectral sort)
        mask_f = mask.float()
        device = mask_f.device
        wave_size = 132
        nnz = mask_f.sum(dim=2)
        n_waves = (MB + wave_size - 1) // wave_size
        base_order = nnz.argsort(dim=1, descending=True)
        pad = n_waves * wave_size - MB
        if pad > 0:
            pad_idx = torch.arange(MB, MB + pad, device=device).unsqueeze(0).expand(H, -1)
            base_order_pad = torch.cat([base_order, pad_idx], dim=1)
            nnz_pad = torch.cat([nnz, torch.zeros(H, pad, dtype=nnz.dtype, device=device)], dim=1)
        else:
            base_order_pad = base_order
            nnz_pad = nnz
        nnz_ordered = torch.gather(nnz_pad, 1, base_order_pad)
        nnz_waves = nnz_ordered.view(H, n_waves, wave_size)
        wave_mean_nnz = nnz_waves.float().mean(dim=2)
        wave_order = wave_mean_nnz.argsort(dim=1, descending=True)
        within_wave_order = nnz_waves.argsort(dim=2, descending=True)
        waves = base_order_pad.view(H, n_waves, wave_size)
        sorted_waves = torch.gather(waves, 2, within_wave_order)
        wave_order_exp = wave_order.unsqueeze(2).expand(H, n_waves, wave_size)
        result_padded = torch.gather(sorted_waves, 1, wave_order_exp)
        valid_mask = result_padded < MB
        return result_padded[valid_mask].view(H, MB).contiguous()
    elif mode == "wave_2d":
        from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import wave_2d_reorder
        return wave_2d_reorder(mask.float(), wave_size=132)
    elif mode == "spectral_2d_only":
        # Stage 1 only with 2D: sort by top-2 singular vectors (angle), no wave scheduling
        mask_f = mask.float()
        device = mask_f.device
        gen = torch.Generator(device=device).manual_seed(42)
        V = torch.randn(H, NB, 2, device=device, generator=gen)
        V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)
        for _ in range(10):
            U = torch.bmm(mask_f, V)
            U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-8)
            V = torch.bmm(mask_f.transpose(1, 2), U)
            V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-8)
        proj = U[:, :, 0:2]
        proj = proj - proj.mean(dim=1, keepdim=True)
        angles = torch.atan2(proj[:, :, 1], proj[:, :, 0])
        return angles.argsort(dim=1)
    else:
        raise ValueError(f"Unknown reorder: {mode}")


def run_fa4(Q, K, V, mask, perm, BLK, B, MB, NB, L, H, D, warmup=2):
    """Build FA4 sparse inputs and run kernel."""
    q_blocked = Q.view(B, MB, BLK, H, D)
    gather_idx = perm.T.unsqueeze(0).unsqueeze(2).unsqueeze(4).expand(B, MB, BLK, H, D)
    q_reordered = torch.gather(q_blocked, 1, gather_idx).view(B, L, H, D).contiguous()

    perm_for_mask = perm.unsqueeze(-1).expand(H, MB, NB)
    reordered_masks = torch.gather(mask, 1, perm_for_mask).bool()
    cnt_all = reordered_masks.sum(dim=-1, dtype=torch.int32)
    sorted_cols = torch.argsort(~reordered_masks, dim=2, stable=True).int()
    col_range = torch.arange(NB, device='cuda').view(1, 1, NB)
    sorted_cols = sorted_cols * (col_range < cnt_all.unsqueeze(-1)).int()

    sp = {
        "full_block_cnt": cnt_all.unsqueeze(0).expand(B, H, MB).contiguous(),
        "full_block_idx": sorted_cols.unsqueeze(0).expand(B, H, MB, NB).contiguous(),
        "mask_block_cnt": torch.zeros(B, H, MB, dtype=torch.int32, device='cuda'),
        "mask_block_idx": torch.zeros(B, H, MB, NB, dtype=torch.int32, device='cuda'),
        "block_size": (BLK, BLK),
    }

    from flash_attn.cute.interface import flash_attn_func as cute_fa
    for _ in range(warmup):
        out = cute_fa(q_reordered, K, V, **sp)
    torch.cuda.synchronize()
    out = cute_fa(q_reordered, K, V, **sp)
    torch.cuda.synchronize()
    return out


def run_single(args):
    """Single mask+reorder — for NCU wrapping."""
    BLK = 128; D = 128; B = 1
    mask = load_mask(args.mask_dir, args.layer, args.threshold)
    H, MB, NB = mask.shape
    L = MB * BLK

    density = mask.float().mean().item()
    nnz = mask.float().sum(dim=2)
    cv = (nnz.std(dim=1, unbiased=False) / nnz.mean(dim=1).clamp(min=1e-8)).mean().item()
    print(f"[CONFIG] mask_dir={os.path.basename(args.mask_dir)} layer={args.layer} "
          f"reorder={args.reorder} L={L} MB={MB} H={H}", file=sys.stderr)
    print(f"[MASK] density={density:.4f} mean_CV={cv:.4f}", file=sys.stderr)

    torch.manual_seed(42)
    Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
    K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
    V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')

    perm = get_reorder_perm(mask, args.reorder)
    run_fa4(Q, K, V, mask, perm, BLK, B, MB, NB, L, H, D)
    print(f"[DONE] layer={args.layer} reorder={args.reorder}", file=sys.stderr)


def find_best_layers(mask_dir, threshold="0_9", top_n=3):
    """Find the top-N highest CV layers in a mask directory."""
    results = []
    for d in sorted(glob.glob(os.path.join(mask_dir, f"mask_*_thr{threshold}"))):
        layer = int(os.path.basename(d).split('_')[1])
        path = os.path.join(d, "mask_no_reorder.pt")
        if not os.path.exists(path):
            continue
        m = torch.load(path, weights_only=False).cuda()
        H, MB, NB = m.shape
        nnz = m.float().sum(dim=2)
        cv = (nnz.std(dim=1, unbiased=False) / nnz.mean(dim=1).clamp(min=1e-8)).mean().item()
        density = m.float().mean().item()
        results.append({"layer": layer, "density": density, "cv": cv, "H": H, "MB": MB})
        del m
    results.sort(key=lambda x: -x["cv"])
    return results[:top_n]


def run_sweep(args):
    """Timing sweep: all sparse_modes × best layers × reorders."""
    BLK = 128; D = 128; B = 1
    threshold = args.threshold.replace('.', '_')
    reorders = ["no_reorder", "nnz_reorder", "wave_1d"]

    from flash_attn.cute.interface import flash_attn_func as cute_fa
    n_warmup = 3; n_repeat = 10

    sparse_modes = sorted(glob.glob(os.path.join(MASK_ROOT, "*")))
    if not sparse_modes:
        print(f"No mask dirs in {MASK_ROOT}. Run paper_dump_masks.sh first.")
        return

    all_results = []

    for sm_dir in sparse_modes:
        sm_name = os.path.basename(sm_dir)
        best = find_best_layers(sm_dir, threshold, top_n=3)
        if not best:
            print(f"[SKIP] {sm_name}: no masks found")
            continue

        print(f"\n{'='*90}")
        print(f"Sparse Mode: {sm_name}")
        print(f"{'='*90}")
        print(f"  Top-3 layers by CV:")
        for b in best:
            print(f"    Layer {b['layer']:2d}: density={b['density']:.4f} CV={b['cv']:.4f} H={b['H']} MB={b['MB']}")

        for layer_info in best[:1]:  # Profile top-1 layer
            layer = layer_info["layer"]
            mask = load_mask(sm_dir, layer, threshold)
            H, MB, NB = mask.shape
            L = MB * BLK

            torch.manual_seed(42)
            Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
            K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
            V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')

            print(f"\n  Profiling Layer {layer} (density={layer_info['density']:.4f} CV={layer_info['cv']:.4f}):")

            for reorder in reorders:
                perm = get_reorder_perm(mask, reorder)

                # Build inputs
                q_blocked = Q.view(B, MB, BLK, H, D)
                gather_idx = perm.T.unsqueeze(0).unsqueeze(2).unsqueeze(4).expand(B, MB, BLK, H, D)
                q_reordered = torch.gather(q_blocked, 1, gather_idx).view(B, L, H, D).contiguous()
                perm_for_mask = perm.unsqueeze(-1).expand(H, MB, NB)
                reordered_masks = torch.gather(mask, 1, perm_for_mask).bool()
                cnt_all = reordered_masks.sum(dim=-1, dtype=torch.int32)
                sorted_cols = torch.argsort(~reordered_masks, dim=2, stable=True).int()
                col_range = torch.arange(NB, device='cuda').view(1, 1, NB)
                sorted_cols = sorted_cols * (col_range < cnt_all.unsqueeze(-1)).int()
                sp = {
                    "full_block_cnt": cnt_all.unsqueeze(0).expand(B, H, MB).contiguous(),
                    "full_block_idx": sorted_cols.unsqueeze(0).expand(B, H, MB, NB).contiguous(),
                    "mask_block_cnt": torch.zeros(B, H, MB, dtype=torch.int32, device='cuda'),
                    "mask_block_idx": torch.zeros(B, H, MB, NB, dtype=torch.int32, device='cuda'),
                    "block_size": (BLK, BLK),
                }

                # Warmup
                for _ in range(n_warmup):
                    out = cute_fa(q_reordered, K, V, **sp)
                    if isinstance(out, tuple): out = out[0]
                torch.cuda.synchronize()

                # Timed
                times = []
                for _ in range(n_repeat):
                    s1 = torch.cuda.Event(enable_timing=True)
                    s2 = torch.cuda.Event(enable_timing=True)
                    s1.record()
                    out = cute_fa(q_reordered, K, V, **sp)
                    s2.record()
                    torch.cuda.synchronize()
                    times.append(s1.elapsed_time(s2))
                attn_ms = sum(times) / len(times)

                result = {
                    "sparse_mode": sm_name, "layer": layer,
                    "density": round(layer_info["density"], 4),
                    "cv": round(layer_info["cv"], 4),
                    "reorder": reorder, "attn_ms": round(attn_ms, 2),
                    "H": H, "MB": MB, "L": L,
                }
                all_results.append(result)
                print(f"    {reorder:18s}  attn={attn_ms:9.2f}ms")

            del Q, K, V, mask
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*90}")
    print(f"SUMMARY: Kernel time (ms) — highest-CV layer per sparse_mode")
    print(f"{'='*90}")
    print(f"{'SparseMode':20s} {'Layer':>5s} {'Density':>8s} {'CV':>6s} {'H':>3s} {'MB':>6s} | "
          + "  ".join(f"{r:>14s}" for r in reorders))
    print("-" * 90)

    by_sm = {}
    for r in all_results:
        key = (r["sparse_mode"], r["layer"])
        if key not in by_sm:
            by_sm[key] = {"info": r}
        by_sm[key][r["reorder"]] = r["attn_ms"]

    for key, data in sorted(by_sm.items()):
        info = data["info"]
        baseline = data.get("no_reorder", 0)
        vals = []
        for reorder in reorders:
            v = data.get(reorder, 0)
            delta = (v - baseline) / baseline * 100 if baseline > 0 else 0
            vals.append(f"{v:7.2f}({delta:+.0f}%)")
        print(f"{info['sparse_mode']:20s} {info['layer']:5d} {info['density']:8.4f} {info['cv']:6.4f} "
              f"{info['H']:3d} {info['MB']:6d} | " + "  ".join(f"{v:>14s}" for v in vals))

    out_path = "paper_timing_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", type=str)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--reorder", type=str, default="no_reorder")
    parser.add_argument("--threshold", type=str, default="0_9")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    elif args.mask_dir:
        run_single(args)
    else:
        print("Specify --mask_dir + --layer for NCU, or --sweep for timing sweep")
