"""Verify required libraries without making optional accelerators mandatory."""
import importlib
import sys

from gpu_compatibility import computation_dtype, supports_fast_attention


REQUIRED_LIBRARIES = (
    "torch", "torchvision", "gradio", "transformers", "diffusers", "cv2", "numpy",
    "wan", "diffsynth",
)
OPTIONAL_LIBRARIES = ("flash_attn", "sageattention", "xformers", "triton")


def verify_install():
    modules = {}
    for name in REQUIRED_LIBRARIES:
        # A broken required dependency must still fail the installation.
        modules[name] = importlib.import_module(name)
        print(f"{name}: {getattr(modules[name], '__version__', 'installed')}")

    torch = modules["torch"]
    print(f"Python {sys.version.split()[0]} | CUDA {torch.version.cuda} | CUDA available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # Importing torch alone does not verify driver or GPU kernel support.
        assert (torch.ones(1, device="cuda") + 1).item() == 2
        print(f"GPU: {torch.cuda.get_device_name()} | compute {torch.cuda.get_device_capability()}")
        print(f"Computation dtype: {computation_dtype()}")

    for name in OPTIONAL_LIBRARIES:
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'installed')} (optional)")
        except Exception as error:
            print(f"{name}: unavailable (optional): {type(error).__name__}: {error}")

    from diffsynth.models import wan_video_dit
    from gpu_compatibility import available_attention_backends
    available = available_attention_backends(wan_video_dit)
    print("Attention backends: " + ", ".join(name for name, enabled in available.items() if enabled))
    if not supports_fast_attention() or not (available["flash"] or available["sage"]):
        print("Using PyTorch SDPA automatically. Optional attention libraries are not required.")
    print("Required libraries verified successfully.")


if __name__ == "__main__":
    verify_install()
