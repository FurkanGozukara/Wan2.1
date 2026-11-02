# V74 attention compatibility validation

Date: 2026-11-03. Baseline: `9e7ad8b` (V73).

## Change

Optional attention DLL import failures no longer fail installation or Wan's
imports. All three installer scripts call the shared `verify_install.py`.
Required dependency failures still fail verification; a small CUDA operation
checks the installed torch kernels when CUDA is available.

The app selects SDPA below compute capability 8.0, including when an old config
requests Sage or Flash. Such GPUs use FP32 computation to preserve the range of
Wan's BF16 weights. Storage precision choices remain available. The native Wan
attention entry points also fall back on missing extensions or unsupported
kernels, preserving padding, scales, bottom-right causal masks, local windows,
grouped query attention and output dtype. OOM and unexpected runtime errors
remain errors. T5's default CUDA device is resolved when instantiated.

## Completed checks

- 27 unittest checks passed in the project's Python 3.12.10 environment, with
  torch 2.13.0+cu130, Gradio 6.29.0, CUDA hidden via `CUDA_VISIBLE_DEVICES=-1`.
  These cover numerical attention definitions, broken imports, unsupported
  kernels, required dependency failures, old/modern hardware selection, saved
  fast choices and the pipeline's actual dtype wiring.
- 18 CUDA checks passed on a cloud RTX A6000 with torch 2.13.0+cu130. FP16 and
  BF16 fast attention matched the baseline bit for bit for dense, padded,
  causal, scaled and local attention. Real Wan cross-attention and CLIP layers
  also matched bit for bit. Missing-extension SDPA and simulated SM 7.5 paths
  matched their SDPA references; the actual layers remained finite and close.
- Full 1.3B generations passed with real DiT, umT5 and VAE weights, first with
  BF16 + Flash and then FP32 + SDPA: 448x256, 17 frames, 24 steps, seed 74,
  CFG 5, FP64 RoPE and zero persistent DiT parameters. Both produced valid
  16 fps MP4 files. Native inspection of decoded middle frames confirmed a
  coherent person and garden in both modes. Four-step structural smoke runs
  also completed before these longer checks.
- A fresh Python subprocess injected the reported `ImportError: DLL load failed
  while importing flash_attn_2_cuda: The specified procedure could not be found.`
  The old installer's verification command exited 1; the new Windows BAT
  verification block exited 0 and reported the exact optional error.
- `Start_APP.bat --no_browser --server_name 127.0.0.1 --server_port 7879` opened
  the V74 UI with CUDA hidden, both normally and with the DLL failure injected.
  Playwright checked the title, attention controls, Models & System tab and
  browser console (zero errors). The owned app/browser processes were closed.
- `bash -n` passed for both Linux installer scripts. The Linux installers were
  syntax checked; a fresh full Linux installation was not repeated.
- The 17-member `Wan21_v74.zip` passed CRC verification. Each updated installer
  matches both its tracked source and its deployed outer-folder copy. Remaining
  members come from the original V73 bundle; requirements are unchanged.

## Reproduce

Use the project's own virtual environment. Hide GPUs for the CPU suite using
`CUDA_VISIBLE_DEVICES=-1`; on PowerShell an empty environment value can remove
the variable instead of hiding the devices.

```text
python -m unittest discover -s tests -p test_attention_fallback.py -v
python verify_install.py
```

After selecting an authorized GPU with `CUDA_VISIBLE_DEVICES`, the separate
CUDA regression command compares against the previous revision:

```text
python tests/check_attention_cuda.py --baseline-ref 9e7ad8b
```

The built archive SHA256 is
`166901a62ac7f143f63561ae2621c5f52e9f4767aee0517283555ebae04957a2`.

## Limits

An actual pre-Ampere GPU was unavailable; SM 7.5 selection was simulated while
executing real CUDA operations on the A6000. The GPU and driver must remain
compatible with the installed PyTorch/CUDA build, and FP32 can need more VRAM.
The report establishes a recoverable optional extension import failure; the
reporter's GPU model and exact binary mismatch are unknown.
