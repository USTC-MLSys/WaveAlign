"""Real pretrained Wan pipeline with the repository's WaveAlign operators.

Diffusers handles model execution; the adapter replaces self-attention with
official XAttention masks and the repository's per-head sparse dispatcher.
No model, weights, attention kernel, or output is mocked.
"""

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers import transformer_wan
from diffusers.utils import export_to_video
from xattn.src.Xattention import xattn_estimate

ROOT = Path(__file__).resolve().parents[2]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_omni/diffusion/attention/backends" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


reorder = load_module("greedy_reorder_cuda")
sparse = load_module("sparse_dispatch")
original_dispatch = transformer_wan.dispatch_attention_fn


class State:
    def __init__(self, args):
        self.args = args
        self.mode = "dense"
        self.layer = -1
        self.step = 0
        self.events = []
        self.densities = []
        self.capture = False
        self.captured = False
        self.capture_seconds = 0.0

    def dispatch(self, q, k, v, **kwargs):
        if self.mode == "dense" or q.shape != k.shape or q.shape[1] <= 4096:
            return original_dispatch(q, k, v, **kwargs)
        batch, length, heads, dim = q.shape
        assert batch == 1, "Wan CFG is processed as separate forwards"
        block = 64
        padded = ((length + block - 1) // block) * block
        padding = padded - length
        qpad = F.pad(q, (0, 0, 0, 0, 0, padding)) if padding else q
        kpad = F.pad(k, (0, 0, 0, 0, 0, padding)) if padding else k
        rows = padded // block
        stamps = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        stamps[0].record()
        _, estimated = xattn_estimate(
            qpad.transpose(1, 2).contiguous(),
            kpad.transpose(1, 2).contiguous(),
            block_size=block,
            stride=16,
            norm=1,
            threshold=self.args.threshold,
            select_mode="inverse",
            use_triton=True,
            causal=False,
            chunk_size=4096,
            kdb=1,
        )
        mask = estimated[0, :, :rows, :rows].bool().contiguous()
        diagonal = torch.arange(rows, device=q.device)
        mask[:, diagonal, diagonal] = True
        stamps[1].record()
        if self.mode == "wave_2d_reorder":
            wave = torch.cuda.get_device_properties(q.device).multi_processor_count * 2
            perm = reorder.wave_2d_reorder(mask.float(), wave_size=wave)
            idx = perm.T[None, :, None, :, None].expand(batch, rows, block, heads, dim)
            qrun = qpad.reshape(batch, rows, block, heads, dim).gather(1, idx).reshape_as(qpad).contiguous()
            selected = mask.gather(1, perm[:, :, None].expand_as(mask))
        else:
            qrun = qpad
            selected = mask
        counts = selected.sum(-1, dtype=torch.int32)[None]
        indices = (~selected).argsort(dim=-1, stable=True).to(torch.int32)[None]
        stamps[2].record()
        result = sparse.sparse_attention(
            qrun,
            k,
            v,
            full_block_cnt=counts,
            full_block_idx=indices,
            block_size=(block, block),
            backend="flashinfer",
            sequential_heads=True,
        )
        stamps[3].record()
        if self.mode == "wave_2d_reorder":
            restored = torch.empty_like(result).reshape(batch, rows, block, heads, dim)
            restored.scatter_(1, idx, result.reshape_as(restored))
            result = restored.reshape_as(result)
        result = result[:, :length]
        stamps[4].record()
        self.events.append(stamps)
        self.densities.append(mask.float().mean().detach())
        if self.capture and not self.captured and self.step == min(15, self.args.steps - 1) and self.layer == 24:
            torch.cuda.synchronize()
            capture_start = time.perf_counter()
            torch.save(
                {
                    "q": q.cpu(),
                    "k": k.cpu(),
                    "v": v.cpu(),
                    "mask": mask.cpu(),
                    "step": self.step,
                    "layer": self.layer,
                    "block": block,
                    "threshold": self.args.threshold,
                },
                self.args.output / "real_qkv_mask.pt",
            )
            self.captured = True
            self.capture_seconds += time.perf_counter() - capture_start
        return result


class Processor:
    def __init__(self, layer, state):
        self.layer, self.state = layer, state
        self.original = transformer_wan.WanAttnProcessor()

    def __call__(self, *args, **kwargs):
        self.state.layer = self.layer
        return self.original(*args, **kwargs)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Local Wan2.1 Diffusers checkpoint directory")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--modes", nargs="+", default=["dense", "no_reorder", "wave_2d_reorder"])
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--save-float-frames", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/wan")
    parser.add_argument(
        "--prompt",
        default=(
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
        ),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("An NVIDIA CUDA GPU is required")
    if args.frames <= 0 or args.frames % 4 != 1 or args.steps <= 0:
        parser.error("Use frames = 4n+1 and a positive step count")
    if any(mode not in {"dense", "no_reorder", "wave_2d_reorder"} for mode in args.modes):
        parser.error("Modes must be dense, no_reorder, or wave_2d_reorder")
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config.update(
        {
            "pipeline": "Diffusers WanPipeline with actual WaveAlign sparse operators",
            "model": str(args.model),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "xattention_revision": "e37988770b9d1bebd489eba011d615f35587ba08",
            "guidance_scale": 6.0,
            "weights": "model loaded in BF16; VAE FP32",
        }
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    print("LOADING PRETRAINED MODEL", flush=True)
    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True).to(
        "cuda"
    )
    pipe.vae.enable_tiling()
    print("MODEL LOADED, allocated GiB=", torch.cuda.memory_allocated() / 2**30, flush=True)
    state = State(args)
    transformer_wan.dispatch_attention_fn = state.dispatch
    for i, block in enumerate(pipe.transformer.blocks):
        block.attn1.set_processor(Processor(i, state))
    negative = (
        "Bright tones, overexposed, static, blurred details, subtitles, worst quality, "
        "low quality, JPEG compression, deformed, disfigured, messy background"
    )
    print("ENCODING PROMPTS", flush=True)
    prompt_embeds, negative_embeds = pipe.encode_prompt(
        prompt=args.prompt, negative_prompt=negative, device="cuda", do_classifier_free_guidance=True
    )
    # Keep pipeline components on CUDA so Diffusers' execution-device lookup
    # remains CUDA after prompt encoding. The A100 has room for the encoder.
    torch.cuda.empty_cache()
    # Compilation and allocator warmup on real model layers at a short shape.
    for mode in ["dense", "no_reorder", "wave_2d_reorder"]:
        state.mode = mode
        state.step = 0
        print("WARMUP", mode, flush=True)
        pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            height=256,
            width=256,
            num_frames=65,
            num_inference_steps=1,
            guidance_scale=6.0,
            generator=torch.Generator("cuda").manual_seed(args.seed),
            output_type="latent",
        )
        state.events = []
        state.densities = []
    metrics = []
    for mode in args.modes:
        state.mode = mode
        state.step = 0
        state.events = []
        state.densities = []
        state.capture_seconds = 0.0
        state.capture = args.capture and mode == "no_reorder"

        def callback(pipeline, index, timestep, values):
            state.step = index + 1
            print("STEP", mode, index + 1, "/", args.steps, flush=True)
            return values

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        print("GENERATE", mode, flush=True)
        latent = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=6.0,
            generator=torch.Generator("cuda").manual_seed(args.seed),
            output_type="latent",
            callback_on_step_end=callback,
        ).frames
        torch.cuda.synchronize()
        denoise = time.perf_counter() - start
        torch.save(latent.cpu(), args.output / f"{mode}_latent.pt")
        sums = [sum(events[i].elapsed_time(events[i + 1]) for events in state.events) for i in range(4)]
        density = torch.stack(state.densities).mean().item() if state.densities else 1.0
        decode_start = time.perf_counter()
        latent = latent.to(pipe.vae.dtype)
        mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda", dtype=latent.dtype).reshape(1, -1, 1, 1, 1)
        std = torch.tensor(pipe.vae.config.latents_std, device="cuda", dtype=latent.dtype).reshape(1, -1, 1, 1, 1)
        decoded = pipe.vae.decode(latent * std + mean, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(decoded, output_type="np")[0]
        torch.cuda.synchronize()
        decode = time.perf_counter() - decode_start
        if args.save_float_frames:
            np.save(args.output / f"{mode}_frames.npy", video)
        else:
            np.save(args.output / f"{mode}_frames.npy", (video.clip(0, 1) * 255).round().astype(np.uint8))
        export_to_video(video, str(args.output / f"{mode}.mp4"), fps=16)
        record = {
            "mode": mode,
            "denoise_seconds": denoise,
            "capture_seconds": state.capture_seconds,
            "denoise_excluding_capture_seconds": denoise - state.capture_seconds,
            "decode_seconds": decode,
            "total_seconds": denoise + decode,
            "sparse_calls": len(state.events),
            "mean_density": density,
            "mask_ms": sums[0],
            "reorder_and_metadata_ms": sums[1],
            "sparse_backend_ms": sums[2],
            "restore_ms": sums[3],
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
            "video_shape": list(video.shape),
            "capture_enabled": state.capture,
        }
        metrics.append(record)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("RESULT", json.dumps(record), flush=True)
        del latent, decoded, video
    comparisons = []
    for left, right in [("no_reorder", "wave_2d_reorder"), ("dense", "no_reorder")]:
        if left not in args.modes or right not in args.modes:
            continue
        a = torch.load(args.output / f"{left}_latent.pt", weights_only=True).float()
        b = torch.load(args.output / f"{right}_latent.pt", weights_only=True).float()
        fa = np.load(args.output / f"{left}_frames.npy", mmap_mode="r")
        fb = np.load(args.output / f"{right}_frames.npy", mmap_mode="r")
        pixel_scale = 255.0 if fa.dtype == np.uint8 else 1.0
        mse = float(np.mean(((fa.astype(np.float64) - fb.astype(np.float64)) / pixel_scale) ** 2))
        comparisons.append(
            {
                "pair": [left, right],
                "latent_exact": torch.equal(a, b),
                "latent_max_abs": (a - b).abs().max().item(),
                "pixel_exact": bool(np.array_equal(fa, fb)),
                "pixel_mse": mse,
                "psnr_db": float(-10 * np.log10(mse)) if mse else "infinity",
            }
        )
    (args.output / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    print("COMPARISONS", json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
