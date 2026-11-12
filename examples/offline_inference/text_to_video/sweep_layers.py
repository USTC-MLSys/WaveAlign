"""Sweep top-5 CV layers per sparse mode — one at a time to avoid OOM."""
import torch, os, glob, sys, gc
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # line-buffered

BLK = 128; D = 128; B = 1
MASK_ROOT = "/data1/dhelix_shared/daizijian/vllm-omni-mask-paper"

print("Loading modules...", flush=True)
from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import wave_overlap_reorder
from flash_attn.cute.interface import flash_attn_func as cute_fa
print("Modules loaded.", flush=True)

def get_perm(mask, reorder, H, MB):
    if reorder == 'nnz_reorder':
        return torch.argsort(mask.float().sum(dim=2), descending=True, dim=1)
    elif reorder == 'wave_1d':
        return wave_overlap_reorder(mask.float(), wave_size=132)
    else:
        return torch.arange(MB, device='cuda').unsqueeze(0).expand(H, MB).contiguous()

def time_kernel(mask, perm, Q, K, V, H, MB, NB, L, n_warmup=2, n_repeat=5):
    q_blocked = Q.view(B, MB, BLK, H, D)
    gi = perm.T.unsqueeze(0).unsqueeze(2).unsqueeze(4).expand(B, MB, BLK, H, D)
    q_r = torch.gather(q_blocked, 1, gi).view(B, L, H, D).contiguous()
    pm = perm.unsqueeze(-1).expand(H, MB, NB)
    rm = torch.gather(mask, 1, pm).bool()
    cnt = rm.sum(dim=-1, dtype=torch.int32)
    sc = torch.argsort(~rm, dim=2, stable=True).int()
    cr = torch.arange(NB, device='cuda').view(1, 1, NB)
    sc = sc * (cr < cnt.unsqueeze(-1)).int()
    sp = {
        'full_block_cnt': cnt.unsqueeze(0).expand(B, H, MB).contiguous(),
        'full_block_idx': sc.unsqueeze(0).expand(B, H, MB, NB).contiguous(),
        'mask_block_cnt': torch.zeros(B, H, MB, dtype=torch.int32, device='cuda'),
        'mask_block_idx': torch.zeros(B, H, MB, NB, dtype=torch.int32, device='cuda'),
        'block_size': (BLK, BLK),
    }
    for _ in range(n_warmup):
        cute_fa(q_r, K, V, **sp)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        s1 = torch.cuda.Event(enable_timing=True)
        s2 = torch.cuda.Event(enable_timing=True)
        s1.record()
        cute_fa(q_r, K, V, **sp)
        s2.record()
        torch.cuda.synchronize()
        times.append(s1.elapsed_time(s2))
    del q_r, gi, pm, rm, cnt, sc, cr
    for k in list(sp.keys()): del sp[k]
    return sum(times)/len(times)

# Header
print(f"{'Mode':20s} {'L':>3s} {'dens':>5s} {'CV':>5s} | {'no':>7s} {'nnz':>7s} {'w1d':>7s} | {'nnz%':>6s} {'w1d%':>6s} {'w1d-nnz':>8s}", flush=True)
print("-"*95, flush=True)

for sm_dir in sorted(glob.glob(os.path.join(MASK_ROOT, '*'))):
    sm = os.path.basename(sm_dir)
    if 'sparge_thr0.5' in sm: continue  # skip old coarsened
    thr = '0_7' if 'sparge' in sm else '0_9'
    
    # First pass: find layers and CV (CPU only)
    layers = []
    for d in sorted(glob.glob(os.path.join(sm_dir, f'mask_*_thr{thr}'))):
        layer = int(os.path.basename(d).split('_')[1])
        p = os.path.join(d, 'mask_no_reorder.pt')
        if not os.path.exists(p): continue
        m = torch.load(p, weights_only=False)
        H, MB, NB = m.shape
        nnz = m.float().sum(dim=2)
        cv = (nnz.std(dim=1) / nnz.mean(dim=1).clamp(min=1e-8)).mean().item()
        density = m.float().mean().item()
        layers.append((layer, density, cv, H, MB, NB))
        del m, nnz
    
    if not layers: continue
    layers.sort(key=lambda x: -x[2])
    
    # Second pass: time top-5 (GPU)
    for layer, density, cv, H, MB, NB in layers[:5]:
        torch.cuda.empty_cache()
        gc.collect()
        
        mask = torch.load(os.path.join(sm_dir, f'mask_{layer}_thr{thr}', 'mask_no_reorder.pt'), weights_only=False).cuda()
        L = MB * BLK
        torch.manual_seed(42)
        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
        
        t_no = time_kernel(mask, get_perm(mask, 'no_reorder', H, MB), Q, K, V, H, MB, NB, L)
        t_nnz = time_kernel(mask, get_perm(mask, 'nnz_reorder', H, MB), Q, K, V, H, MB, NB, L)
        t_w1d = time_kernel(mask, get_perm(mask, 'wave_1d', H, MB), Q, K, V, H, MB, NB, L)
        
        d_nnz = (t_nnz - t_no)/t_no*100
        d_w1d = (t_w1d - t_no)/t_no*100
        w_vs_n = (t_w1d - t_nnz)/t_nnz*100
        star = ' ★' if w_vs_n < -2.0 else (' ◆' if w_vs_n < -1.0 else '')
        
        print(f"{sm:20s} L{layer:2d} {density:5.3f} {cv:5.3f} | {t_no:7.1f} {t_nnz:7.1f} {t_w1d:7.1f} | {d_nnz:+5.1f}% {d_w1d:+5.1f}% {w_vs_n:+6.1f}%{star}", flush=True)
        
        del Q, K, V, mask
        torch.cuda.empty_cache()
        gc.collect()

print("\nDone!", flush=True)
