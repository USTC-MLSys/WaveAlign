import os
import re
import matplotlib.pyplot as plt

log_dir = './output_sparge_tp2_dy'
# modes = ['no_reorder', 'nnz_reorder', 'bcg_reorder', 'opt_reorder', 'wave_reorder']  # 'fast_greedy' 暂时不用
modes = ['no_reorder',  'wave_reorder', 'wave_2d_reorder']  # 'fast_greedy' 暂时不用

def extract_frame(filename):
    m = re.search(r'_f(\d+)', filename)
    return int(m.group(1)) if m else None

def extract_threshold(filename):
    m = re.search(r'thr([\d.]+)', filename)
    return m.group(1) if m else None

def extract_main_time(filepath):
    # 匹配所有 100%|...| 2/2 [01:42<00:00, 51.05s/it]，取最后一个的时间
    with open(filepath, 'r') as f:
        lines = f.readlines()
    times = []
    for line in lines:
        m = re.search(r'^100%\s*\|\s*█+\s*\|\s*2/2\s*\[\d{2}:\d{2}<\d{2}:\d{2},\s*([\d.]+)s/it\]$', line)
        if m:
            times.append(float(m.group(1)))
    return times[-1] if times else None

def extract_attn_avg(filepath):
    # 匹配所有 [MY_LOG] 总计算子 (Pure Attn): ... ms，取平均
    with open(filepath, 'r') as f:
        text = f.read()
    ms = [float(x) for x in re.findall(r'总计算子 \\(Pure Attn\\): ([\d.]+) ms', text)]
    if not ms:
        ms = [float(x) for x in re.findall(r'总计算子 \\(Pure Attn\\): ([\d.]+) ms', text.replace('（', '(').replace('）', ')'))]
    if not ms:
        ms = [float(x) for x in re.findall(r'Pure Attn\): ([\d.]+) ms', text)]
    return sum(ms)/len(ms)/1000 if ms else None  # 转为秒

# 新增：提取重排(reorder)均值，单位转为秒
def extract_reorder_avg(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    ms = [float(x) for x in re.findall(r'总计重排 \\(Reorder\\): ([\d.]+) ms', text)]
    if not ms:
        ms = [float(x) for x in re.findall(r'总计重排 \\(Reorder\\): ([\d.]+) ms', text.replace('（', '(').replace('）', ')'))]
    if not ms:
        ms = [float(x) for x in re.findall(r'Reorder\): ([\d.]+) ms', text)]
    return sum(ms)/len(ms)/1000 if ms else None  # 转为秒

# 收集数据: data[threshold][mode][frame] = (main_time, attn_avg, reorder_avg)
data = {}
for fname in os.listdir(log_dir):
    if not fname.endswith('.log'):
        continue
    frame = extract_frame(fname)
    threshold = extract_threshold(fname)
    if threshold is None or frame is None:
        continue
    for mode in modes:
        if fname.startswith(f'vllm_{mode}'):
            main_time = extract_main_time(os.path.join(log_dir, fname))
            attn_avg = extract_attn_avg(os.path.join(log_dir, fname))
            reorder_avg = extract_reorder_avg(os.path.join(log_dir, fname))
            if main_time is not None and attn_avg is not None and reorder_avg is not None:
                if threshold not in data:
                    data[threshold] = {m: {} for m in modes}
                data[threshold][mode][frame] = (main_time, attn_avg, reorder_avg)
            else:
                print(f"no data {threshold=} {mode=} {frame=}")

# 画图: 每个threshold一个小图，四根线
# 画图
num_thresholds = len(data)
fig, axes = plt.subplots(2, num_thresholds, figsize=(6*num_thresholds, 10), squeeze=False)
# 第一排：加速比
frame2seqlen = {161:147, 241:220, 321:291, 401:363, 481:435, 521:471, 561:508, 601:544, 641:580, 681:616, 721:651, 801:723, 881:795}
for idx, (threshold, mode_dict) in enumerate(sorted(data.items(), key=lambda x: float(x[0]))):
    ax = axes[0, idx]
    frames = sorted(set.intersection(*(set(mode_dict[m].keys()) for m in modes)))
    seqlens = [frame2seqlen.get(f, f) for f in frames]
    seqlen_labels = [str(s) for s in seqlens]
    def get_speedup(mode, ref_mode, frames, idx_):
        return [mode_dict[ref_mode][f][idx_] / mode_dict[mode][f][idx_] if f in mode_dict[mode] and f in mode_dict[ref_mode] else None for f in frames]
    # speedup_nnz = get_speedup('nnz_reorder', 'no_reorder', frames, 0)
    # speedup_bcg = get_speedup('bcg_reorder', 'no_reorder', frames, 0)
    # speedup_opt = get_speedup('opt_reorder', 'no_reorder', frames, 0)
    speedup_wave = get_speedup('wave_reorder', 'no_reorder', frames, 0)
    # speedup_wave_1d = get_speedup('wave_1d_reorder', 'no_reorder', frames, 0)
    speedup_wave_2d = get_speedup('wave_2d_reorder', 'no_reorder', frames, 0)
    # speedup_fast = get_speedup('fast_greedy', 'no_reorder', frames, 0)

    # speedup_nnz_attn = get_speedup('nnz_reorder', 'no_reorder', frames, 1)
    # speedup_bcg_attn = get_speedup('bcg_reorder', 'no_reorder', frames, 1)
    # speedup_opt_attn = get_speedup('opt_reorder', 'no_reorder', frames, 1)
    speedup_wave_attn = get_speedup('wave_reorder', 'no_reorder', frames, 1)
    # speedup_wave_1d_attn = get_speedup('wave_1d_reorder', 'no_reorder', frames, 1)
    speedup_wave_2d_attn = get_speedup('wave_2d_reorder', 'no_reorder', frames, 1)
    # speedup_fast_attn = get_speedup('fast_greedy', 'no_reorder', frames, 1)

    # ax.plot(seqlens, speedup_nnz_attn, marker='^', label='nnz_reorder/Attn')
    # ax.plot(seqlens, speedup_bcg_attn, marker='v', label='bcg_reorder/Attn')
    # ax.plot(seqlens, speedup_opt_attn, marker='v', label='opt_reorder/Attn')
    ax.plot(seqlens, speedup_wave_attn, marker='v', label='wave_reorder/Attn')
    # ax.plot(seqlens, speedup_wave_1d_attn, marker='v', label='wave_1d_reorder/Attn')
    ax.plot(seqlens, speedup_wave_2d_attn, marker='v', label='wave_2d_reorder/Attn')
    # ax.plot(seqlens, speedup_fast_attn, marker='x', label='fast_greedy/Attn')

    # ax.plot(seqlens, speedup_nnz, marker='o', label='nnz_reorder/Main')
    # ax.plot(seqlens, speedup_bcg, marker='D', label='bcg_reorder/Main')
    # ax.plot(seqlens, speedup_opt, marker='D', label='opt_reorder/Main')
    ax.plot(seqlens, speedup_wave, marker='D', label='wave_reorder/Main')
    # ax.plot(seqlens, speedup_wave_1d, marker='D', label='wave_1d_reorder/Main')
    ax.plot(seqlens, speedup_wave_2d, marker='D', label='wave_2d_reorder/Main')
    # ax.plot(seqlens, speedup_fast, marker='s', label='fast_greedy/Main')
    ax.set_xlabel('seqlen(K)')
    ax.set_ylabel('Speedup')
    ax.set_title(f'Threshold=Dynamic')
    ax.set_xticks(seqlens)
    ax.set_xticklabels(seqlen_labels)
    ax.legend(loc='lower left')

# 第二排：重排时间占比（不画no_reorder）
for idx, (threshold, mode_dict) in enumerate(sorted(data.items(), key=lambda x: float(x[0]))):
    ax = axes[1, idx]
    frames = sorted(set.intersection(*(set(mode_dict[m].keys()) for m in modes)))
    seqlens = [frame2seqlen.get(f, f) for f in frames]
    seqlen_labels = [str(s) for s in seqlens]
    for mode in modes:
        if mode == 'no_reorder':
            continue
        reorder_ratios = []
        for f in frames:
            tup = mode_dict[mode][f]
            reorder_avg = tup[2]
            main_time = tup[0]
            reorder_ratios.append((reorder_avg / 2) / main_time if main_time else None)
        ax.plot(seqlens, reorder_ratios, marker='o', label=mode)
    ax.set_xlabel('seqlen(K)')
    ax.set_ylabel('Reorder/Main Time Ratio')
    ax.set_title(f'Threshold=Dynamic Reorder Ratio')
    ax.set_xticks(seqlens)
    ax.set_xticklabels(seqlen_labels)
    ax.legend()
# # 第二排：attn两个值的差值占较大值的比重
# for idx, (threshold, mode_dict) in enumerate(sorted(data.items(), key=lambda x: float(x[0]))):
#     ax = axes[1, idx]
#     frames = sorted(set.intersection(*(set(mode_dict[m].keys()) for m in modes)))
#     seqlens = [frame2seqlen.get(f, f) for f in frames]
#     seqlen_labels = [str(s) for s in seqlens]
#     for mode in modes:
#         diffs = []
#         for f in frames:
#             # 读取该mode下该frame的两个attn值
#             fname = None
#             for fn in os.listdir(log_dir):
#                 if fn.startswith(f'vllm_{mode}') and f'_f{f}' in fn and f'thr{threshold}' in fn:
#                     fname = fn
#                     break
#             attn_vals = []
#             if fname:
#                 with open(os.path.join(log_dir, fname), 'r') as ff:
#                     text = ff.read()
#                 attn_vals = [float(x) for x in re.findall(r'Pure Attn\): ([\d.]+) ms', text)]
#                 if not attn_vals:
#                     attn_vals = [float(x) for x in re.findall(r'Pure Attn): ([\d.]+) ms', text)]
#             if len(attn_vals) >= 2:
#                 diff = abs(attn_vals[0] - attn_vals[1]) / max(attn_vals)
#                 diffs.append(diff)
#             else:
#                 diffs.append(None)
#         ax.plot(seqlens, diffs, marker='o', label=mode)
#     ax.set_xlabel('seqlen')
#     ax.set_ylabel('Attn Diff Ratio')
#     ax.set_title(f'Threshold={threshold} Attn Diff Ratio')
#     ax.set_xticks(seqlens)
#     ax.set_xticklabels(seqlen_labels)
#     ax.legend()
plt.tight_layout()
plt.savefig('speedup_vllm.png')