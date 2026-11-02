Wan 2.1 Amazing APP only for https://www.patreon.com/posts/123105403

V74 automatically uses PyTorch SDPA when accelerated attention cannot load or
the GPU has compute capability below 8.0. GPUs without native BF16 support use
FP32 computation, which can require more VRAM. Modern GPUs keep the existing
Sage/Flash attention choices and BF16 computation.

FlashAttention, SageAttention, xFormers and Triton are optional accelerators.
The installer reports their import errors and continues when required libraries
work. A DLL import error alone does not establish that a GPU is too old. The GPU
and driver must still be supported by the installed PyTorch/CUDA build.

Use the V74 installer bundle, or replace the installer in the existing outer
installation folder with the matching file from `installers/`, then run it.
Keep `requirements.txt`, `uv_build_constraints.txt` and the download scripts next
to that installer. The checkout remains in the `Wan2.1/` subfolder.

Regression checks (run with the app's virtual environment):

```text
python -m unittest discover -s tests -p test_attention_fallback.py -v
python verify_install.py
```

With an authorized CUDA GPU selected, run the comparison against V73 with
`python tests/check_attention_cuda.py --baseline-ref 9e7ad8b`.
See [the validation report](docs/validation_v74.md) for results and limits.

Build the updated bundle from the previous installer archive:

```text
python tools/package_release.py --base-zip ../Wan21_v73.zip --output ../Wan21_v74.zip
```
