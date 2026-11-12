# batch_test.sh 参数说明

本文说明 [batch_test.sh](batch_test.sh) 中各参数的作用，以及任务是如何被组织和分派到 GPU 上的。

## 运行方式

在 [vllm-omni/examples/offline_inference/text_to_video](vllm-omni/examples/offline_inference/text_to_video) 目录下执行：

```bash
bash batch_test.sh
```

如果你当前不在这个目录，也可以用绝对路径执行：

```bash
bash /data1/dhelix_shared/daizijian/vllm-omni/examples/offline_inference/text_to_video/batch_test.sh
```

推荐先进入 `vllm-omni` 对应环境，再运行脚本。脚本本身已经设置了 `LD_LIBRARY_PATH`，但 Python 解释器、依赖包和 CUDA 相关环境仍然需要和当前实验环境一致。

## 脚本用途

[batch_test.sh](batch_test.sh) 用于批量执行 `text_to_video.py`，对不同的：

- `FRAMES_LIST`
- `THRESHOLDS`
- `MODES`

组合逐个跑推理任务，并把结果视频和日志分别输出到：

- `output_mp4/`
- `output_${SPARSE_MODE}_tp${TP}/`

如果某个任务对应的日志文件已经存在，脚本会直接跳过，避免重复执行。

## 当前任务遍历顺序

脚本当前生成任务的顺序是：

1. `FRAMES_LIST` 最外层
2. `THRESHOLDS` 中间层
3. `MODES` 最内层

也就是同一个帧数下，会依次遍历所有阈值；在同一个阈值下，再依次遍历所有 `mode`。

## 主要参数

### 稀疏相关参数

#### `SPARSE_MODE`

控制稀疏 mask 的生成方式。当前代码里常见的两种取值：

- `xattn`
- `sparge`

它会通过环境变量传给推理代码：

```bash
SPARSE_MODE=$SPARSE_MODE VLLM_THRESHOLD=$THR VLLM_MODE=$MODE python text_to_video.py ...
```

#### `THRESHOLDS`

这是脚本里最容易混淆的参数。`THRESHOLDS` 的含义取决于 `SPARSE_MODE`。

当 `SPARSE_MODE=xattn` 时：

- `THRESHOLDS` 对应 `VLLM_THRESHOLD`
- 它表示基于块级 `attnscore` 累计和的保留阈值
- 更准确地说，是“累计注意力和占总和的比例阈值”

实现上，`xattn_estimate` 会先得到每个块的 `attn_sum`，随后在 `find_blocks_chunked` 中按从大到小排序，选择足够多的块，使其累计和达到：

$$
\text{required\_sum} = \text{total\_sum} \times \text{threshold}
$$

所以：

- 阈值越大，通常会保留更多块
- 阈值越小，通常会更稀疏

例如：

- `0.3` 表示累计覆盖到总 `attnscore` 的 30%
- `0.7` 表示累计覆盖到总 `attnscore` 的 70%

当 `SPARSE_MODE=sparge` 时：

- `THRESHOLDS` 同样通过 `VLLM_THRESHOLD` 传入
- 但它不再表示 `attnscore` 的累计和阈值
- 它会被传给 `get_block_map_meansim(..., simthreshd1=threshold, ...)`
- 这里的含义是 `simthreshold`，也就是相似度阈值

换句话说：

- `xattn` 模式下，`THRESHOLDS` 是 `attnscore` 求和的 threshold
- `sparge` 模式下，`THRESHOLDS` 是 `simthreshold`

补充一点：当前 `sparge` 路径里还固定传了 `topk=0.1`，因此最终 mask 不只受 `simthreshold` 影响，也会受这个固定 top-k 筛选比例影响。

#### `MODES`

控制重排或调度策略，通过环境变量 `VLLM_MODE` 传入后端。

例如当前脚本中包含：

- `no_reorder`
- `nnz_reorder`
- `wave_reorder`
- `wave_1d_reorder`
- `wave_2d_reorder`

这些模式用于改变 block mask 或 block 行的组织顺序，从而影响访存局部性和整体性能。

### 视频生成参数

#### `FRAMES_LIST`

待测试的视频帧数列表。脚本会对每个帧数都跑一整套 `threshold × mode` 组合。

#### `HEIGHT` 和 `WIDTH`

输出视频分辨率。当前配置：

- `HEIGHT=1280`
- `WIDTH=720`

#### `STEPS`

扩散采样步数，对应 `--num-inference-steps`。数值越大，一般生成时间越长。

### 并行和 GPU 参数

#### `TP`

Tensor Parallel 大小，对应命令行参数：

```bash
--tensor-parallel-size $TP
```

当前设置为 `TP=2`，表示每个推理任务会占用 2 张卡。

#### `GPU_IDS`

可用于运行任务的 GPU 编号列表，例如：

```bash
GPU_IDS=(0 1 2 3 4 5 6 7)
```

脚本不会自动探测空闲卡，而是严格按这里给出的编号分组使用。

#### `GPU_COUNT`

由 `GPU_IDS` 自动计算得到：

```bash
GPU_COUNT=${#GPU_IDS[@]}
```

#### `GROUP_COUNT`

表示当前最多可同时跑多少个任务：

$$
\text{GROUP\_COUNT} = \frac{\text{GPU\_COUNT}}{\text{TP}}
$$

例如当前：

- `GPU_COUNT=8`
- `TP=2`
- 所以 `GROUP_COUNT=4`

即最多并发 4 个任务。

### 其他运行参数

#### `LD_LIBRARY_PATH`

脚本开头显式把 `xattn` 环境下的库目录加入 `LD_LIBRARY_PATH`，避免运行时找不到动态链接库。

#### `TUNE_MODE=0`

关闭调优模式，按固定配置执行。

#### `TORCH_COMPILE_DISABLE=1`

禁用 `torch.compile`，通常是为了减少编译开销和避免不稳定因素。

## 输出文件命名

输出视频文件名格式：

```text
output_mp4/vllm_${MODE}_thr${THR}_f${FRAMES}.mp4
```

日志文件名格式：

```text
output_${SPARSE_MODE}_tp${TP}_dy/vllm_${MODE}_thr${THR}_f${FRAMES}.log
```

因此一个任务的关键信息都体现在文件名里：

- 稀疏模式体现在日志目录名中
- 重排策略体现在 `${MODE}`
- 阈值体现在 `${THR}`
- 帧数体现在 `${FRAMES}`

## 使用建议

### 切换到 `xattn` 时

如果把：

```bash
SPARSE_MODE=xattn
```

那么建议把 `THRESHOLDS` 理解为“注意力累计覆盖率阈值”，而不是相似度阈值。此时常见调参思路是：

- 想更稀疏：减小 `THRESHOLDS`
- 想保留更多注意力块：增大 `THRESHOLDS`

### 切换到 `sparge` 时

如果把：

```bash
SPARSE_MODE=sparge
```

那么建议把 `THRESHOLDS` 理解为 `simthreshold`。这时它控制的是块相似性筛选，而不是 `attnscore` 累积覆盖率。

## 一句话总结

- `xattn` 下，`THRESHOLDS` 是块级 `attnscore` 累计和的 threshold
- `sparge` 下，`THRESHOLDS` 是 `simthreshold`
- `TP` 决定每个任务占几张卡
- `GPU_IDS` 决定实际用哪些 GPU
- 已存在日志的任务会自动跳过