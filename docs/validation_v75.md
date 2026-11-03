# V75 installer environment validation

Date: 2026-11-03. Baseline: `e266f65` (V74).

## Change

The reported installation log installed pip and uv in `Wan2.1/venv`, but both
uv package operations selected `C:\Python312`. The model downloader then failed
to import `requests` from the app's venv, even though it is a requirement.
The log establishes the environment mismatch; it does not identify the
reporter's inherited settings.

All three installers now run pip, uv, the downloader and verification through
the app's explicit venv interpreter. Both uv package operations also pass that
interpreter with `--python`. Linux venv paths are absolute so changing directory
for requirements or editable DiffSynth installation cannot change the target.
The app version is V75. Dependencies and generation behavior are unchanged.

## Completed checks

- A fresh Python 3.12.10 venv without `requests` was created in a directory
  containing spaces. With the reported uv 0.13.7, inherited `UV_PYTHON` pointing
  at system Python and `UV_SYSTEM_PYTHON=true`, the old uv command selected
  system Python in a dry run. No system packages were changed.
- The patched Windows installer's actual uv bootstrap and both dependency
  command lines ran through CMD against that venv. A focused requirements file
  installed `requests` and `truststore`; the real DiffSynth package built and
  installed editable with `--no-deps` and the existing build constraints.
  Both uv operations selected the venv. The unchanged downloader modules then
  imported successfully, with `requests` loaded from the fresh venv.
- `bash -n` passed for both Linux installer scripts. Their deployed outer-folder
  copies match the tracked sources byte for byte.
- The unchanged `Start_APP.bat` launcher opened the V75 app with CUDA hidden.
  Playwright confirmed the V75 heading and page title, opened Models & System,
  and reported zero browser console errors or warnings. Only the task-owned
  app and browser processes were stopped after verification.
- `Wan21_v75.zip` passed CRC verification. All 17 members were checked against
  V74; installer scripts and requirements match the tracked sources and
  deployed outer-folder copies, and all other members are byte-identical to
  V74. The original V74 archive remains unchanged.

## Build

Run with the app's own virtual environment:

```text
python tools/package_release.py --base-zip ../Wan21_v74.zip --output ../Wan21_v75.zip
```

Archive size: 598002 bytes. SHA256:
`57f310a3670854d8d2c0da42837acbfac19fb82d42169133862a7749b20791bd`.

## Limits

This was a focused fresh-venv dependency regression check, not a full fresh
installation of every GPU dependency. Linux installers were syntax checked;
a full Linux installation was not repeated. Generation was not repeated
because no generation code or dependencies changed.
