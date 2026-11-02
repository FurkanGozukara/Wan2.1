"""Hardware checks shared by startup, installation and attention dispatch."""
import torch


def supports_fast_attention(device=None):
    if not torch.cuda.is_available():
        return False
    device = torch.device(device) if device is not None else torch.device("cuda")
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def computation_dtype(device="cuda"):
    # Wan's BF16 weights can exceed FP16's range. FP32 keeps that range on older
    # GPUs; switching the entire pipeline to FP16 can produce non-finite output.
    return torch.bfloat16 if supports_fast_attention(device) else torch.float32


def available_attention_backends(dit, device=None):
    available = dit.available_attention_backends().copy()
    if not supports_fast_attention(device):
        available.update(sage=False, flash=False)
    return available
