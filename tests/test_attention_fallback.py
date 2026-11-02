"""Numerical and failure-path checks; no CUDA GPU or model downloads needed."""
import builtins
import contextlib
import importlib.util
import io
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

import gpu_compatibility as compat
import verify_install


ROOT = Path(__file__).resolve().parents[1]


def load_attention(import_error=None):
    spec = importlib.util.spec_from_file_location("attention_under_test", ROOT / "wan/modules/attention.py")
    module = importlib.util.module_from_spec(spec)
    original = builtins.__import__

    def controlled_import(name, *args, **kwargs):
        if name in ("flash_attn", "flash_attn_interface"):
            raise (import_error or ModuleNotFoundError)("DLL load failed: The specified procedure could not be found.")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=controlled_import):
        spec.loader.exec_module(module)
    return module


def reference(q, k, v, q_lens=None, k_lens=None, scale=None, q_scale=1., causal=False, window=(-1, -1)):
    """Explicit per-head FP64 definition, independent of SDPA."""
    b, lq, nq, _ = q.shape
    out = torch.zeros(b, lq, nq, v.size(-1), dtype=q.dtype)
    for i in range(b):
        qlen = lq if q_lens is None else q_lens[i]
        klen = k.size(1) if k_lens is None else k_lens[i]
        for h in range(nq):
            if not qlen or not klen:
                continue
            kh = h // (nq // k.size(2))
            scores = (q[i, :qlen, h].double() * q_scale) @ k[i, :klen, kh].double().T
            scores *= scale if scale is not None else q.size(-1) ** -0.5
            for row in range(qlen):
                center = row + klen - qlen
                for col in range(klen):
                    if (causal and col > center) or (window[0] >= 0 and col < center - window[0]) or (window[1] >= 0 and col > center + window[1]):
                        scores[row, col] = -torch.inf
            probabilities = torch.softmax(scores, dim=-1).nan_to_num()
            out[i, :qlen, h] = (probabilities @ v[i, :klen, kh].double()).to(q.dtype)
    return out


class AttentionFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attn = load_attention()

    def setUp(self):
        torch.manual_seed(73)
        self.q = torch.randn(2, 5, 4, 8)
        self.k = torch.randn(2, 7, 4, 8)
        self.v = torch.randn(2, 7, 4, 6)
        self.attn._failed_backends.clear()

    def check_reference(self, **kwargs):
        expected_args = {k: kwargs[k] for k in ("q_lens", "k_lens", "causal") if k in kwargs}
        expected_args.update(scale=kwargs.get("softmax_scale"), q_scale=kwargs.get("q_scale", 1.), window=kwargs.get("window_size", (-1, -1)))
        expected = reference(self.q, self.k, self.v, **expected_args)
        for entry in (self.attn.flash_attention, self.attn.attention):
            actual = entry(self.q, self.k, self.v, **kwargs)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            self.assertEqual(actual.dtype, self.q.dtype)

    def test_missing_or_broken_extensions_do_not_prevent_import(self):
        for error in (ModuleNotFoundError, ImportError, OSError, RuntimeError):
            with self.subTest(error=error):
                module = load_attention(error)
                self.assertFalse(module.FLASH_ATTN_2_AVAILABLE)
                self.assertFalse(module.FLASH_ATTN_3_AVAILABLE)
                self.assertTrue(torch.isfinite(module.flash_attention(self.q, self.k, self.v)).all())

    def test_dense(self):
        self.check_reference()

    def test_query_and_softmax_scales(self):
        self.check_reference(softmax_scale=0.31, q_scale=1.7)

    def test_padding_is_excluded(self):
        self.k[0, 3:] = 10000
        self.v[0, 3:] = 10000
        self.check_reference(k_lens=torch.tensor([3, 7]))

    def test_ragged_queries_and_keys(self):
        self.check_reference(q_lens=[2, 4], k_lens=[4, 6], q_scale=0.7)

    def test_causal_unequal_lengths(self):
        self.check_reference(causal=True)

    def test_causal_more_queries_than_keys(self):
        self.q, self.k = self.k, self.q
        self.v = self.v[:, :5]
        self.check_reference(causal=True)

    def test_causal_equal_lengths(self):
        self.k, self.v = self.k[:, :5], self.v[:, :5]
        self.check_reference(causal=True)

    def test_local_window(self):
        self.check_reference(window_size=(2, 1))

    def test_local_causal_ragged(self):
        self.check_reference(window_size=(2, 0), causal=True, q_lens=[3, 5], k_lens=[6, 4])

    def test_grouped_query_attention(self):
        self.k, self.v = self.k[:, :, :2], self.v[:, :, :2]
        self.check_reference(k_lens=[4, 7])

    def test_empty_keys_and_queries(self):
        self.check_reference(q_lens=[0, 5], k_lens=[7, 0])

    def test_invalid_lengths_fail(self):
        for lengths in ([8, 7], [-1, 3], [3], [2.5, 3]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                self.attn.flash_attention(self.q, self.k, self.v, k_lens=lengths)

    def test_fp16_dtype_and_values(self):
        self.q, self.k, self.v = (x.half() for x in (self.q, self.k, self.v))
        actual = self.attn.flash_attention(self.q, self.k, self.v)
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual, reference(self.q, self.k, self.v), atol=1e-3, rtol=1e-3)

    def test_dropout_is_forwarded(self):
        fn = torch.nn.functional.scaled_dot_product_attention
        with patch.object(torch.nn.functional, "scaled_dot_product_attention", wraps=fn) as call:
            self.attn.flash_attention(self.q, self.k, self.v, dropout_p=0.2)
            self.assertEqual(call.call_args.kwargs["dropout_p"], 0.2)

    def test_working_fast_backend_is_preserved(self):
        expected = self.q.clone()
        with patch.object(self.attn, "_flash_versions", return_value=[2]), patch.object(self.attn, "_flash_attention", return_value=expected) as fast:
            self.assertIs(self.attn.flash_attention(self.q, self.k, self.v), expected)
            self.assertEqual(fast.call_args.kwargs["version"], 2)

    def test_unsupported_kernel_falls_back_and_is_recorded(self):
        with patch.object(self.attn, "_flash_versions", return_value=[2]), patch.object(self.attn, "_flash_attention", side_effect=RuntimeError("no kernel image is available for execution on the device")):
            self.check_reference()
        self.assertIn((self.q.device, 2), self.attn._failed_backends)

    def test_oom_and_unexpected_errors_are_not_hidden(self):
        for error in (torch.OutOfMemoryError("out of memory"), RuntimeError("CUDA error: an illegal memory access was encountered"), RuntimeError("invalid tensor shape")):
            with self.subTest(error=error), patch.object(self.attn, "_flash_versions", return_value=[2]), patch.object(self.attn, "_flash_attention", side_effect=error), self.assertRaises(type(error)):
                self.attn.flash_attention(self.q, self.k, self.v)
        self.assertFalse(self.attn._failed_backends)


class HardwareTests(unittest.TestCase):
    def test_older_gpu_uses_sdpa_and_fp32(self):
        dit = types.SimpleNamespace(available_attention_backends=lambda: dict(sage=True, flash=True, sdpa=True))
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
            self.assertEqual(compat.computation_dtype(), torch.float32)
            self.assertEqual(compat.available_attention_backends(dit), dict(sage=False, flash=False, sdpa=True))

    def test_modern_gpu_keeps_existing_backends_and_bf16(self):
        dit = types.SimpleNamespace(available_attention_backends=lambda: dict(sage=True, flash=True, sdpa=True))
        for cap in ((8, 0), (8, 6), (8, 9), (9, 0), (12, 0)):
            with self.subTest(cap=cap), patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_capability", return_value=cap):
                self.assertEqual(compat.computation_dtype(), torch.bfloat16)
                self.assertTrue(compat.available_attention_backends(dit)["sage"])

    def test_cpu_does_not_query_cuda_device(self):
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("must not access a CUDA GPU")):
            self.assertFalse(compat.supports_fast_attention())
            self.assertEqual(compat.computation_dtype(), torch.float32)

    def test_gpu_dispatch_skips_older_architectures(self):
        attn = load_attention()
        q = types.SimpleNamespace(device=torch.device("cuda:0"), size=lambda dim: 128)
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)), patch.object(attn, "FLASH_ATTN_2_AVAILABLE", True), patch.object(attn, "FLASH_ATTN_3_AVAILABLE", True):
            self.assertEqual(attn._flash_versions(q, None), [])


class InstallerTests(unittest.TestCase):
    def test_optional_dll_errors_are_reported_and_installation_succeeds(self):
        original = verify_install.importlib.import_module

        def broken_optional(name, *args, **kwargs):
            if name in verify_install.OPTIONAL_LIBRARIES:
                raise ImportError("DLL load failed: The specified procedure could not be found.")
            return original(name, *args, **kwargs)

        output = io.StringIO()
        with patch.object(verify_install.importlib, "import_module", side_effect=broken_optional), contextlib.redirect_stdout(output):
            verify_install.verify_install()
        self.assertIn("flash_attn: unavailable (optional): ImportError: DLL load failed", output.getvalue())
        self.assertIn("Required libraries verified successfully.", output.getvalue())

    def test_required_dependency_errors_still_fail(self):
        with patch.object(verify_install.importlib, "import_module", side_effect=ImportError("broken torch")), self.assertRaisesRegex(ImportError, "broken torch"):
            verify_install.verify_install()

    def test_real_app_imports_without_initializing_cuda(self):
        # T5's default device used to initialize CUDA at import time.
        with patch.object(torch.cuda, "current_device", side_effect=AssertionError("eager CUDA initialization")), patch.object(torch.cuda, "is_available", return_value=False):
            import App
            self.assertEqual(App.APP_VERSION, "V74")
            self.assertEqual(App.default_attention_choice(), App.ATTENTION_SDPA)
            self.assertEqual(App.apply_attention_choice(App.ATTENTION_SAGE), "sdpa")

    def test_saved_fast_choice_falls_back_on_older_gpu(self):
        import App
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
            self.assertEqual(App.default_attention_choice(), App.ATTENTION_SDPA)
            self.assertEqual(App.apply_attention_choice(App.ATTENTION_FLASH), "sdpa")

    def test_pipeline_uses_compatible_computation_dtype(self):
        import App
        for cap, expected in (((7, 5), torch.float32), ((8, 6), torch.bfloat16)):
            with self.subTest(cap=cap), patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_capability", return_value=cap), patch.object(App, "ensure_model_downloaded"), patch.object(App, "get_common_file", side_effect=lambda a, b: a), patch.object(App, "ModelManager") as manager, patch.object(App.WanVideoPipeline, "from_model_manager") as pipeline:
                App.load_wan_pipeline("1.3B", "torch.bfloat16", "0")
                self.assertEqual(pipeline.call_args.kwargs["torch_dtype"], expected)
                manager.return_value.load_models.assert_called_once()
                pipeline.return_value.enable_vram_management.assert_called_once_with(num_persistent_param_in_dit=0)


if __name__ == "__main__":
    unittest.main()
