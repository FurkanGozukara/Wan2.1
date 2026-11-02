# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import warnings

from gpu_compatibility import supports_fast_attention

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except Exception as error:
    FLASH_ATTN_3_AVAILABLE = False
    warnings.warn(f'FlashAttention 3 unavailable; using another attention backend: {error}')

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except Exception as error:
    FLASH_ATTN_2_AVAILABLE = False
    warnings.warn(f'FlashAttention 2 unavailable; using PyTorch SDPA when needed: {error}')

_failed_backends = set()

__all__ = [
    'flash_attention',
    'attention',
]


def _flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if version == 3:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def _sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """SDPA with the same lengths, bottom-right causal mask and scales as FA2."""
    out_dtype = q.dtype
    work_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else dtype
    if q.device.type == 'cuda' and not supports_fast_attention(q.device) and work_dtype == torch.bfloat16:
        work_dtype = torch.float32
    q, k, v = (x.transpose(1, 2).to(work_dtype) for x in (q, k, v))
    if q_scale is not None:
        q = q * q_scale

    b, _, lq, _ = q.shape
    lk = k.size(2)

    def lengths(values, maximum):
        if values is None:
            return [maximum] * b
        values = torch.as_tensor(values).tolist()
        if len(values) != b or any(int(n) != n or n < 0 or n > maximum for n in values):
            raise ValueError('Attention lengths must contain one valid sequence length per batch item.')
        return [int(n) for n in values]

    q_lengths, k_lengths = lengths(q_lens, lq), lengths(k_lens, lk)

    def run(qi, ki, vi):
        nq, nk = qi.size(2), ki.size(2)
        if not nq or not nk:
            return qi.new_zeros(qi.size(0), qi.size(1), nq, vi.size(-1))
        mask = None
        simple_causal = causal and nq == nk and window_size == (-1, -1)
        if (causal and not simple_causal) or window_size != (-1, -1):
            # FlashAttention aligns unequal query/key lengths at the bottom right.
            row = torch.arange(nq, device=q.device)[:, None] + nk - nq
            col = torch.arange(nk, device=q.device)[None, :]
            mask = torch.ones(nq, nk, dtype=torch.bool, device=q.device)
            if causal:
                mask &= col <= row
            if window_size[0] >= 0:
                mask &= col >= row - window_size[0]
            if window_size[1] >= 0:
                mask &= col <= row + window_size[1]
        return torch.nn.functional.scaled_dot_product_attention(
            qi, ki, vi, attn_mask=mask, dropout_p=dropout_p,
            is_causal=simple_causal, scale=softmax_scale,
            enable_gqa=qi.size(1) != ki.size(1))

    if q_lens is None and k_lens is None:
        out = run(q, k, v)
    else:
        # Trim each sequence rather than allocating a quadratic padding mask.
        out = q.new_zeros(b, q.size(1), lq, v.size(-1))
        for i, (nq, nk) in enumerate(zip(q_lengths, k_lengths)):
            out[i:i + 1, :, :nq] = run(q[i:i + 1, :, :nq], k[i:i + 1, :, :nk], v[i:i + 1, :, :nk])
    return out.transpose(1, 2).contiguous().to(out_dtype)


def _flash_versions(q, version):
    if q.device.type != 'cuda' or q.size(-1) > 256 or not supports_fast_attention(q.device):
        return []
    versions = []
    # The separately installed FA3 extension targets Hopper, not every CUDA GPU.
    if version in (None, 3) and FLASH_ATTN_3_AVAILABLE and torch.cuda.get_device_capability(q.device)[0] == 9:
        versions.append(3)
    if FLASH_ATTN_2_AVAILABLE:
        versions.append(2)
    return [v for v in versions if (q.device, v) not in _failed_backends]


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                    softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False,
                    dtype=torch.bfloat16, version=None):
    kwargs = dict(q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
                  softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
                  window_size=window_size, deterministic=deterministic, dtype=dtype)
    # The original packed FA path requires equal query lengths for unflatten().
    for selected in _flash_versions(q, version) if q_lens is None else []:
        try:
            return _flash_attention(q, k, v, version=selected, **kwargs)
        except torch.OutOfMemoryError:
            raise
        except (ImportError, OSError, NotImplementedError) as error:
            reason = str(error)
        except RuntimeError as error:
            # Do not hide OOM, invalid inputs, or a poisoned CUDA context.
            if not any(term in str(error).lower() for term in (
                'not supported', 'only support', 'unsupported', 'no kernel image',
                'invalid device function', 'not compiled', 'requires', 'undefined symbol',
            )):
                raise
            reason = str(error)
        _failed_backends.add((q.device, selected))
        warnings.warn(f'FlashAttention {selected} cannot run on {q.device} ({reason}); falling back automatically.')
    return _sdpa_attention(q, k, v, **kwargs)


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
              softmax_scale=None, q_scale=None, causal=False,
              window_size=(-1, -1), deterministic=False,
              dtype=torch.bfloat16, fa_version=None):
    return flash_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
                           q_scale, causal, window_size, deterministic, dtype, fa_version)
