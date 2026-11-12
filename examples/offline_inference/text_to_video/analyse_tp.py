import re

def analyze_tp_latency(file_rank0, file_rank1):
    def parse_file(file_path):
        latencies = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    # 使用正则提取 ms 数值
                    match = re.search(r"Latency:\s+([\d.]+)\s+ms", line)
                    if match:
                        latencies.append(float(match.group(1)))
        except FileNotFoundError:
            print(f"错误: 找不到文件 {file_path}")
        return latencies

    lat0 = parse_file(file_rank0)
    lat1 = parse_file(file_rank1)

    # 确保行数对齐
    if not lat0 or not lat1:
        print("其中一个文件为空或未找到有效数据。")
        return

    min_len = min(len(lat0), len(lat1))
    if len(lat0) != len(lat1):
        print(f"警告: 行数不一致 (Rank0: {len(lat0)}, Rank1: {len(lat1)})，将只统计前 {min_len} 行。")

    # 1. 计算各自总耗时
    sum0 = sum(lat0[:min_len])
    sum1 = sum(lat1[:min_len])

    # 2. 计算每一行对应的最大值之和 (TP 实际阻塞后的总时长)
    max_step_latencies = [max(lat0[i], lat1[i]) for i in range(min_len)]
    total_max_sum = sum(max_step_latencies)

    # 3. 计算两卡的平均总耗时
    avg_sum = (sum0 + sum1) / 2

    # 4. 计算并行效率 (Average / TotalMaxSum)
    parallel_efficiency = (avg_sum / total_max_sum) * 100 if total_max_sum > 0 else 0

    print("="*60)
    print(f" TP=2 负载均衡深度分析报告 (对齐行数: {min_len})")
    print("-" * 60)
    print(f"Rank 0 累计计算耗时:    {sum0:>12.2f} ms")
    print(f"Rank 1 累计计算耗时:    {sum1:>12.2f} ms")
    print(f"两卡平均计算耗时:       {avg_sum:>12.2f} ms")
    print("-" * 60)
    print(f"实际执行总开销 (Max Sum): {total_max_sum:>12.2f} ms")
    print(f"并行效率 (Avg/MaxSum):  {parallel_efficiency:>11.2f} %")
    print("-" * 60)
    
    # 诊断建议
    if parallel_efficiency < 95:
        print("!! 诊断: 并行效率较低。存在明显的负载倾斜 (Skew)，")
        print("   这通常是由于 Sparse Attention 的 Mask 在两卡上分配不均导致的。")
    else:
        print("√ 诊断: 负载非常均衡，两卡计算进度高度同步。")
    print("="*60)

# 指定你的日志文件路径
analyze_tp_latency("nnz_attn_latency_rank_0.log", "nnz_attn_latency_rank_1.log")