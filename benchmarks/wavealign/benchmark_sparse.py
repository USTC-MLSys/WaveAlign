"""Synthetic-mask measurements using the pinned per-head FlashInfer fork."""

import argparse
import gc
import random

import flashinfer
import torch
from common import cuda_timing, dump_json, make_mask, policies, reorder_q, restore_output, verify_permutation


def csr(mask):
    ptrs, columns = [], []
    for head in mask:
        counts = head.sum(-1, dtype=torch.int32)
        ptrs.append(torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32))))
        columns.append(head.nonzero()[:, 1].to(torch.int32).contiguous())
    return ptrs, columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768, 131072, 436096])
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["no_reorder", "nnz_reorder", "wave_1d_reorder", "wave_2d_reorder", "svd_u2u3_reorder"],
    )
    parser.add_argument("--prefix", default="benchmark")
    parser.add_argument("--block-size", type=int, choices=[64, 128], default=64)
    parser.add_argument("--wave-size", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("An NVIDIA CUDA GPU is required")
    if not hasattr(flashinfer, "PerHeadBlockSparseAttentionWrapper"):
        parser.error("Install the pinned FlashInfer fork described in third_party/flashinfer/README.md")
    block, dim, heads = args.block_size, 128, args.heads
    if heads <= 0 or any(length <= 0 or length % block for length in args.lengths):
        parser.error("Use positive head counts and lengths divisible by block size")
    wave = args.wave_size or torch.cuda.get_device_properties(0).multi_processor_count * (2 if block == 64 else 1)
    if wave <= 0:
        parser.error("Wave size must be positive")
    methods = args.methods
    results = []
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "fork_commit": "9215ce86df41f9b9005eb0c34ff23bdc281327a7",
        "heads": heads,
        "dim": dim,
        "block": block,
        "wave": wave,
        "dtype": "bfloat16",
        "masks": "synthetic, not captured from a video model",
        "kernel": "prepared per-head FA2 calls; excludes plan and layout copies",
        "full": "mask provided -> reorder -> Q/mask gather -> CSR -> plan -> wrapper.run -> restore",
    }
    dump_json(args.prefix + "_environment.json", env)
    for length in args.lengths:
        rows = length // block
        gen = torch.Generator(device="cuda").manual_seed(length)
        q = torch.randn(heads, length, dim, device="cuda", dtype=torch.bfloat16, generator=gen)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for pattern, density in [("random", 0.5), ("clustered", 0.35), ("ordered_clusters", 0.35)]:
            mask = make_mask(heads, rows, rows, pattern=pattern, density=density, seed=length + 1)
            workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            wrapper = flashinfer.PerHeadBlockSparseAttentionWrapper(workspace, backend="fa2")

            def plan(selected):
                ptrs, cols = csr(selected)
                wrapper.plan(
                    ptrs,
                    cols,
                    length,
                    length,
                    block,
                    block,
                    heads,
                    heads,
                    dim,
                    sequential_heads=True,
                    q_data_type=q.dtype,
                    kv_data_type=q.dtype,
                    o_data_type=q.dtype,
                )

            def run(query):
                return wrapper.run(
                    query.transpose(0, 1).contiguous(), k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()
                ).transpose(0, 1)

            plan(mask)
            baseline = run(q).contiguous()
            torch.cuda.synchronize()
            order = methods.copy()
            random.Random(length + rows).shuffle(order)
            for name in order:
                op = policies(wave)[name]
                perm = op(mask)
                verify_permutation(perm, mask)
                selected = mask.gather(1, perm[..., None].expand_as(mask))
                query = reorder_q(q[None], perm, block)[0] if name != "no_reorder" else q
                plan(selected)
                actual = run(query)
                if name != "no_reorder":
                    actual = restore_output(actual[None], perm, block)[0]
                error = (actual - baseline).abs().max().item()
                torch.testing.assert_close(actual, baseline, atol=0.016, rtol=0.016)
                # Prepared kernel arguments exclude per-head layout conversion.
                prepared = [
                    (query[h, :, None].contiguous(), k[h, :, None].contiguous(), v[h, :, None].contiguous())
                    for h in range(heads)
                ]

                def kernels():
                    return tuple(w.run(*tensors) for w, tensors in zip(wrapper._sequential_wrappers, prepared))

                repeat = 5 if length >= 131072 else 9
                kernel_time = cuda_timing(kernels, warmup=2, repeat=repeat)
                algo_time = cuda_timing(lambda: op(mask), warmup=1, repeat=5)
                plan_time = cuda_timing(lambda: plan(selected), warmup=1, repeat=3)

                def full():
                    if name == "no_reorder":
                        plan(mask)
                        return run(q)
                    p = op(mask)
                    selected_mask = mask.gather(1, p[..., None].expand_as(mask))
                    query_reordered = reorder_q(q[None], p, block)[0]
                    plan(selected_mask)
                    out = run(query_reordered)
                    return restore_output(out[None], p, block)[0]

                complete = cuda_timing(full, warmup=1, repeat=repeat)
                # Workload proxy, not a measured L2 hit rate.
                union_counts = [selected[:, i : i + wave].any(1).sum(1).float() for i in range(0, rows, wave)]
                union_mean = torch.stack(union_counts).mean().item()
                record = {
                    "length": length,
                    "pattern": pattern,
                    "actual_density": mask.float().mean().item(),
                    "method": name,
                    "max_abs_error": error,
                    "kernel": kernel_time,
                    "reorder": algo_time,
                    "plan": plan_time,
                    "full": complete,
                    "mean_wave_kv_union_blocks": union_mean,
                    "max_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
                results.append(record)
                dump_json(args.prefix + "_results.json", results)
                print(
                    f"{length:7d} {pattern:16s} {name:22s} kernel={kernel_time['gpu_median_ms']:9.3f} "
                    f"full={complete['wall_median_ms']:9.3f} algo={algo_time['gpu_median_ms']:8.3f} error={error:.6f}",
                    flush=True,
                )
                actual = query = selected = prepared = None
            baseline = mask = wrapper = workspace = None
            gc.collect()
        q = k = v = None
        torch.cuda.empty_cache()
    print("Completed", len(results), "measurements", flush=True)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
