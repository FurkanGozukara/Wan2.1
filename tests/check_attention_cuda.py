"""Compare actual CUDA attention and model callers with a previous revision."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-ref', default='9e7ad8b')
parser.add_argument('--baseline-file', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
if not torch.cuda.is_available():
    raise SystemExit('A CUDA GPU is required. Select the authorized GPU with CUDA_VISIBLE_DEVICES before running.')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

new = load('candidate_attention', root / 'wan/modules/attention.py')
source = args.baseline_file.read_text() if args.baseline_file else subprocess.check_output(['git', 'show', f'{args.baseline_ref}:wan/modules/attention.py'], cwd=root, text=True)
old = types.ModuleType('baseline_attention')
exec(compile(source, 'baseline_attention.py', 'exec'), old.__dict__)
results = {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'baseline': args.baseline_ref, 'checks': []}
torch.manual_seed(74)
for dtype in (torch.bfloat16, torch.float16):
    q = torch.randn(2, 129, 4, 64, device='cuda', dtype=dtype)
    k = torch.randn(2, 157, 4, 64, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    for kwargs in ({}, {'k_lens': torch.tensor([101, 157], device='cuda')}, {'causal': True}, {'softmax_scale': .17, 'q_scale': .7}, {'window_size': (5, 2)}):
        before = old.flash_attention(q, k, v, **kwargs)
        after = new.flash_attention(q, k, v, **kwargs)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        results['checks'].append({'modern_fast_path': str(dtype), 'arguments': list(kwargs), 'bit_identical': True})
    with patch.object(new, 'FLASH_ATTN_2_AVAILABLE', False), patch.object(new, 'FLASH_ATTN_3_AVAILABLE', False):
        after = new.flash_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)).transpose(1,2).contiguous()
    torch.testing.assert_close(after, expected, atol=0, rtol=0)
    results['checks'].append({'no_extensions': str(dtype), 'sdpa_bit_identical': True})
    with patch.object(torch.cuda, 'get_device_capability', return_value=(7, 5)), patch.object(new, '_flash_attention', side_effect=AssertionError('must skip unsupported kernel')):
        after = new.flash_attention(q, k, v)
    work = torch.float32 if dtype == torch.bfloat16 else dtype
    expected = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).to(work), k.transpose(1,2).to(work), v.transpose(1,2).to(work)).transpose(1,2).contiguous().to(dtype)
    torch.testing.assert_close(after, expected, atol=0, rtol=0)
    results['checks'].append({'simulated_sm75': str(dtype), 'finite': bool(torch.isfinite(after).all()), 'sdpa_bit_identical': True})

# Exercise the real Wan DiT and CLIP callers as well as the public function.
import wan.modules.model as model
import wan.modules.clip as clip
for module, layer, x, call in (
    (model, model.WanT2VCrossAttention(128, 4, (-1, -1)), torch.randn(1, 16, 128), lambda layer, x: layer(x, torch.randn(1, 9, 128, device=x.device, dtype=x.dtype), torch.tensor([7], device=x.device))),
    (clip, clip.SelfAttention(128, 4, causal=True), torch.randn(1, 16, 128), lambda layer, x: layer(x)),
):
    layer = layer.cuda().bfloat16().eval()
    x = x.cuda().bfloat16()
    torch.manual_seed(5)
    with torch.no_grad(), patch.object(module, 'flash_attention', old.flash_attention):
        before = call(layer, x)
    torch.manual_seed(5)
    with torch.no_grad(), patch.object(module, 'flash_attention', new.flash_attention):
        after = call(layer, x)
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    results['checks'].append({'caller': type(layer).__name__, 'modern_bit_identical': True})
    torch.manual_seed(5)
    with torch.no_grad(), patch.object(torch.cuda, 'get_device_capability', return_value=(7, 5)), patch.object(module, 'flash_attention', new.flash_attention):
        after = call(layer, x)
    assert torch.isfinite(after).all()
    torch.testing.assert_close(after, before, atol=0.02, rtol=0.05)
    results['checks'].append({'caller': type(layer).__name__, 'simulated_sm75_finite_and_close': True})
torch.cuda.synchronize()
if args.output:
    args.output.write_text(json.dumps(results, indent=2))
print(f"{len(results['checks'])} CUDA regression checks passed on {results['gpu']} ({results['torch']}).")
