"""
Paper micro-benchmark figure: 5 sparse modes × 3 reorder strategies.
Top row: L2 Hit Rate + DRAM Read (from NCU)
Bottom row: Kernel Time (from timing sweep)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
# Format: (display_name, layer, density, cv, threshold,
#          {reorder: (l2_hit, dram_gb, attn_ms)})

DATA = [
    ("XAttn",       78, 0.579, 0.87, "0.9", {
        "no_reorder":  (47.73, 2901.4, 1774.0),
        "nnz_reorder": (71.83, 1404.7, 1654.0),
        "wave_1d":     (85.57,  660.6, 1590.0),
    }),
    ("SpargeAttn",   3, 0.385, 0.33, "0.7", {
        "no_reorder":  (31.44, 2808.0, 1238.0),
        "nnz_reorder": (83.36,  481.2, 1070.0),
        "wave_1d":     (88.85,  313.4, 1059.0),
    }),
    ("LongCat",     41, 0.559, 0.73, "0.9", {
        "no_reorder":  (53.72, 2377.0, 1674.0),
        "nnz_reorder": (86.13,  591.8, 1551.0),
        "wave_1d":     (88.83,  468.2, 1537.0),
    }),
    ("BSA",          2, 0.695, 0.52, "0.9", {
        "no_reorder":  (44.02, 3858.3, 2143.0),
        "nnz_reorder": (85.69,  768.0, 1933.0),
        "wave_1d":     (87.54,  628.4, 1933.0),
    }),
    ("vMoBA",        0, 0.977, 0.03, "0.9", {
        "no_reorder":  (26.89, 7759.1, 3200.0),
        "nnz_reorder": (55.21, 3900.4, 2925.0),
        "wave_1d":     (55.80, 3991.4, 2875.0),
    }),
]

reorder_labels = ["no_reorder", "nnz_reorder", "wave_1d"]
reorder_display = ["No Reorder", "NNZ Reorder", "Wave-1D"]
colors = ["#E74C3C", "#3498DB", "#2ECC71"]  # red, blue, green

mode_names = [d[0] for d in DATA]
n = len(mode_names)
x = np.arange(n)
width = 0.25

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Micro-Benchmark: Row Reordering Impact on FA4 Sparse Attention\n"
             "(BLK=128, TP=1, H=40, seqlen≈292K, Wan2.2-14B)",
             fontsize=14, fontweight='bold')

# X-axis labels: Mode (Layer, ρ=density, CV)
xlabels = []
for d in DATA:
    name, layer, density, cv, thr, _ = d
    xlabels.append(f"{name}\n(L{layer}, ρ={density:.2f})")

# Subplot 1: L2 Hit Rate
ax = axes[0]
for i, (ro, label, color) in enumerate(zip(reorder_labels, reorder_display, colors)):
    vals = [d[5][ro][0] for d in DATA]
    bars = ax.bar(x + i*width - width, vals, width, label=label, color=color, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax.set_ylabel('L2 Hit Rate (%)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylim(0, 108)
ax.legend(fontsize=8, loc='upper left')
ax.set_title('L2 Cache Hit Rate', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Subplot 2: DRAM Read (GB)
ax = axes[1]
for i, (ro, label, color) in enumerate(zip(reorder_labels, reorder_display, colors)):
    vals = [d[5][ro][1] for d in DATA]
    bars = ax.bar(x + i*width - width, vals, width, label=label, color=color, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{v:.0f}', ha='center', va='bottom', fontsize=6, fontweight='bold', rotation=45)
ax.set_ylabel('DRAM Read (GB)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=8)
ax.legend(fontsize=8, loc='upper right')
ax.set_title('DRAM Traffic', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Subplot 3: Kernel Time (ms)
ax = axes[2]
for i, (ro, label, color) in enumerate(zip(reorder_labels, reorder_display, colors)):
    vals = [d[5][ro][2] for d in DATA]
    bars = ax.bar(x + i*width - width, vals, width, label=label, color=color, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{v:.0f}', ha='center', va='bottom', fontsize=6, fontweight='bold', rotation=45)
ax.set_ylabel('Kernel Time (ms)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=8)
ax.legend(fontsize=8, loc='upper right')
ax.set_title('FA4 Kernel Latency', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('paper_microbench_5modes.pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper_microbench_5modes.png', bbox_inches='tight', dpi=200)
print("Saved paper_microbench_5modes.pdf/.png")

# ── Print summary improvement table ──
print("\n" + "="*90)
print("IMPROVEMENT SUMMARY (wave_1d vs no_reorder)")
print("="*90)
print(f"{'Mode':12s} {'L':>4s} {'Dens':>5s} {'CV':>5s} | "
      f"{'L2: no→wave':>14s} | {'DRAM: no→wave':>20s} | {'Time: no→wave':>20s}")
print("-"*90)
for name, layer, density, cv, thr, results in DATA:
    nr = results["no_reorder"]
    w = results["wave_1d"]
    l2_delta = w[0] - nr[0]
    dram_delta = (w[1] - nr[1]) / nr[1] * 100
    time_delta = (w[2] - nr[2]) / nr[2] * 100
    print(f"{name:12s} L{layer:>2d} {density:5.2f} {cv:5.2f} | "
          f"{nr[0]:5.1f}→{w[0]:5.1f}% (+{l2_delta:.0f}pp) | "
          f"{nr[1]:7.0f}→{w[1]:5.0f}GB ({dram_delta:+.0f}%) | "
          f"{nr[2]:7.0f}→{w[2]:5.0f}ms ({time_delta:+.0f}%)")
