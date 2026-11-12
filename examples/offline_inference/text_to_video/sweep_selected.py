"""Sweep selected layers to find best wave_1d vs nnz differentiation."""
import torch, os, gc, json, sys
sys.stdout.reconfigure(line_buffering=True)

BLK = 128; D = 128; B = 1
MASK_ROOT = "/data1/dhelix_shared/daizijian/vllm-omni-mask-paper"

from vllm_omni.diffusion.attention.backends.greedy_reorder_cuda import wave_overlap_reorder
from flash_attn.cute.interface import flash_attn_func as cute_fa

def get_perm(mask, reorder, H, MB):
    if reorder == 'nnz_reorder':
        return torch.argsort(mask.float().sum(dim=2), descending=True, dim=1)
    elif reorder == 'wave_1d':
        return wave_overlap_reorder(mask.float(), wave_size=132)
    return torch.arange(MB, device='cuda').unsqueeze(0).expand(H, MB).contiguous()

def run_one(mask, perm, Q, K, V, H, MB, NB, L):
    q_b = Q.view(B, MB, BLK, H, D)
    gi = perm.T.unsqueeze(0).unsqueeze(2).unsqueeze(4).expand(B, MB, BLK, H, D)
    q_r = torch.gather(q_b, 1, gi).view(B, L, H, D).contiguous()
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
    for _ in range(3):
        cute_fa(q_r, K, V, **sp)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        s1 = torch.cuda.Event(enable_timing=True)
        s2 = torch.cuda.Event(enable_timing=True)
        s1.record(); cute_fa(q_r, K, V, **sp); s2.record()
        torch.cuda.synchronize(); times.append(s1.elapsed_time(s2))
    return sum(times)/len(times)

test_cases = [
    ('xattn_thr0.9', 78, '0_9'),
    ('xattn_thr0.9', 38, '0_9'),
    ('sparge_thr0.7', 3, '0_7'),
    ('longcat_cdf_thr0.9', 41, '0_9'),
    ('bsa_cdf_thr0.9', 2, '0_9'),
    ('vmoba_thr_thr0.9', 0, '0_9'),
]

print("Starting sweep...", flush=True)
results = []
for sm_name, layer, thr in test_cases:
    torch.cuda.empty_cache(); gc.collect()
    path = os.path.join(MASK_ROOT, sm_name, f'mask_{layer}_thr{thr}', 'mask_no_reorder.pt')
    if not os.path.exists(path):
        print(f'{sm_name} L{layer} SKIP', flush=True)
        continue
    print(f'Loading {sm_name} L{layer}...', flush=True)
    mask = torch.load(path, weights_only=False, map_location='cpu').cuda()
    H, MB, NB = mask.shape
    nnz = mask.float().sum(dim=2)
    cv = (nnz.std(dim=1) / nnz.mean(dim=1).clamp(min=1e-8)).mean().item()
    density = mask.float().mean().item()
    L = MB * BLK
    
    torch.manual_seed(42)
    Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
    K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
    V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device='cuda')
    
    print(f'  Running no_reorder...', flush=True)
    t_no = run_one(mask, get_perm(mask, 'no_reorder', H, MB), Q, K, V, H, MB, NB, L)
    print(f'  Running nnz_reorder...', flush=True)
    t_nnz = run_one(mask, get_perm(mask, 'nnz_reorder', H, MB), Q, K, V, H, MB, NB, L)
    print(f'  Running wave_1d...', flush=True)
    t_w1d = run_one(mask, get_perm(mask, 'wave_1d', H, MB), Q, K, V, H, MB, NB, L)
    
    w_vs_n = (t_w1d - t_nnz)/t_nnz*100
    star = ' ★' if w_vs_n < -2 else ''
    print(f'  {sm_name:20s} L{layer:2d} d={density:.3f} CV={cv:.3f} | no={t_no:.0f} nnz={t_nnz:.0f} w1d={t_w1d:.0f} | w1d-nnz={w_vs_n:+.1f}%{star}', flush=True)
    results.append({'mode': sm_name, 'layer': layer, 'density': round(density,4), 'cv': round(cv,4), 'no': round(t_no,1), 'nnz': round(t_nnz,1), 'w1d': round(t_w1d,1)})
    del Q, K, V, mask, nnz

json.dump(results, open('sweep_selected_results.json', 'w'), indent=2)
print('DONE', flush=True)
