#!/usr/bin/env python3
"""
Parse MFU benchmark logs and produce a comparison table.

Usage:
    python parse_mfu_logs.py [log_dir]
    
Default log_dir: mfu_logs/
"""

import os
import re
import sys
import glob


def parse_log(filepath: str) -> dict:
    """Extract key metrics from a single benchmark log.
    
    TP=2 produces two reports (one per rank) with potentially interleaved output.
    We average all matches for each metric across ranks.
    """
    result = {
        "mode": os.path.basename(filepath).replace("mfu_", "").replace(".log", ""),
        "file": filepath,
    }
    
    patterns = {
        "total_sparse_ms":   r"\[MY_LOG\] 总计稀疏 \(Sparse\):\s+([\d.]+) ms",
        "total_reorder_ms":  r"\[MY_LOG\] 总计重排 \(Reorder\):\s+([\d.]+) ms",
        "algo_ms":           r"\[MY_LOG\]\s+├─ algo:\s+([\d.]+) ms",
        "q_gather_ms":       r"\[MY_LOG\]\s+├─ Q_gather:\s+([\d.]+) ms",
        "sparse_ms":         r"\[MY_LOG\]\s+├─ sparse:\s+([\d.]+) ms",
        "expand_ms":         r"\[MY_LOG\]\s+└─ expand:\s+([\d.]+) ms",
        "total_attn_ms":     r"\[MY_LOG\] 总计算子 \(Pure Attn\):\s+([\d.]+) ms",
        "total_o_shuffle_ms":r"\[MY_LOG\] 总计O_shuffle \(O_shuffle\):\s+([\d.]+) ms",
        "count":             r"\[MY_LOG\] 总计count\s+(\d+)",
        "avg_density":       r"\[MFU\] Avg block density:\s+([\d.]+)",
        "total_dense_flops": r"\[MFU\] Total dense FLOPs:\s+([\d.eE+]+)",
        "total_sparse_flops":r"\[MFU\] Total sparse FLOPs:\s+([\d.eE+]+)",
        "attn_kernel_ms":    r"\[MFU\] Attn kernel time:\s+([\d.]+) ms",
        "avg_call_attn_ms":  r"\[MFU\] Avg per-call latency:\s+([\d.]+) ms",
        "mfu_pct":           r"\[MFU\] MFU:\s+([\d.]+)%",
        "achieved_tflops":   r"\[MFU\] Achieved TFLOPS/s:\s+([\d.]+)",
        "eff_tflops":        r"\[MFU\] Eff\. TFLOPS/s:\s+([\d.]+)",
        "eff_speedup":       r"\[MFU\] Eff\. Speedup vs dense:\s+([\d.]+)",
        "e2e_time":          r"Total generation time:\s+([\d.]+) seconds",
    }
    
    with open(filepath, "r") as f:
        content = f.read()
    
    for key, pat in patterns.items():
        matches = re.findall(pat, content)
        if matches:
            vals = [float(m) for m in matches]
            # Average across all ranks
            result[key] = sum(vals) / len(vals)
    
    return result


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "mfu_logs"
    
    if not os.path.isdir(log_dir):
        print(f"[ERROR] Log directory not found: {log_dir}")
        sys.exit(1)
    
    logs = sorted(glob.glob(os.path.join(log_dir, "mfu_*.log")))
    if not logs:
        print(f"[ERROR] No mfu_*.log files found in {log_dir}/")
        sys.exit(1)
    
    results = [parse_log(f) for f in logs]
    
    # Print comparison table
    print("\n" + "=" * 110)
    print(f"{'Reorder Mode':<20} {'Attn(ms)':>10} {'Reorder(ms)':>12} {'Total(ms)':>10} "
          f"{'Density':>8} {'MFU%':>8} {'TFLOPS/s':>10} {'Eff.Speedup':>12} {'E2E(s)':>8}")
    print("-" * 110)
    
    for r in results:
        attn = r.get("total_attn_ms", 0)
        reorder = r.get("total_reorder_ms", 0)
        total = attn + reorder
        density = r.get("avg_density", 0)
        mfu = r.get("mfu_pct", 0)
        tflops = r.get("achieved_tflops", 0)
        speedup = r.get("eff_speedup", 0)
        e2e = r.get("e2e_time", 0)
        
        print(f"{r['mode']:<20} {attn:>10.1f} {reorder:>12.1f} {total:>10.1f} "
              f"{density:>8.4f} {mfu:>8.2f} {tflops:>10.2f} {speedup:>12.2f}× {e2e:>8.2f}")
    
    print("=" * 110)
    
    # Speedup vs no_reorder baseline
    baseline = next((r for r in results if r["mode"] == "no_reorder"), None)
    if baseline and len(results) > 1:
        base_attn = baseline.get("total_attn_ms", 1)
        base_total = base_attn + baseline.get("total_reorder_ms", 0)
        print(f"\n{'─'*60}")
        print(f"  Speedup vs no_reorder (attn kernel only / attn+reorder)")
        print(f"{'─'*60}")
        for r in results:
            if r["mode"] == "no_reorder":
                continue
            attn = r.get("total_attn_ms", 1)
            total = attn + r.get("total_reorder_ms", 0)
            attn_speedup = base_attn / attn if attn > 0 else 0
            total_speedup = base_total / total if total > 0 else 0
            mfu_gain = r.get("mfu_pct", 0) - baseline.get("mfu_pct", 0)
            print(f"  {r['mode']:<20}  attn: {attn_speedup:.3f}x  |  attn+reorder: {total_speedup:.3f}x  |  MFU Δ: {mfu_gain:+.2f}%")
        print(f"{'─'*60}")
    
    # Detailed breakdown per mode
    print(f"\n{'='*80}")
    print(f"  Detailed Reorder Breakdown")
    print(f"{'='*80}")
    for r in results:
        mode = r['mode']
        print(f"\n  [{mode}]")
        print(f"    Sparse estimation:  {r.get('total_sparse_ms', 0):>8.1f} ms")
        print(f"    Reorder total:      {r.get('total_reorder_ms', 0):>8.1f} ms")
        print(f"      algo:             {r.get('algo_ms', 0):>8.1f} ms")
        print(f"      Q_gather:         {r.get('q_gather_ms', 0):>8.1f} ms")
        print(f"      sparse build:     {r.get('sparse_ms', 0):>8.1f} ms")
        print(f"      expand:           {r.get('expand_ms', 0):>8.1f} ms")
        print(f"    Attn kernel:        {r.get('total_attn_ms', 0):>8.1f} ms")
        print(f"    O_shuffle:          {r.get('total_o_shuffle_ms', 0):>8.1f} ms")
        print(f"    Count:              {r.get('count', 0):>8.0f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
