"""Thin shim that maps the real flash-attn public API onto vLLM's already-installed,
ABI-compatible flash-attention kernels (``vllm.vllm_flash_attn``).

NOTE ON CORRECTNESS: vLLM ships a *forward-only* flash-attn build (its ``.so`` reports
"This flash attention build does not support backward." and exposes only ``varlen_fwd``).
Therefore these wrappers are numerically correct for the forward pass only. They are NOT
differentiable and MUST NOT be used on a path that backpropagates through attention.
"""

from vllm.vllm_flash_attn import flash_attn_varlen_func as _vllm_varlen_func


def _map_window_size(window_size):
    # real flash-attn: (-1, -1) == full attention; vLLM: None == full attention.
    if window_size is None:
        return None
    if tuple(window_size) == (-1, -1):
        return None
    return [int(window_size[0]), int(window_size[1])]


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    **kwargs,
):
    """Real flash-attn ``flash_attn_varlen_func`` signature, delegating to vLLM by keyword.

    Argument NAMES/order differ from vLLM's function, so everything is passed by keyword.
    """
    win = _map_window_size(window_size)
    out = _vllm_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=win,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        block_table=block_table,
        return_softmax_lse=bool(return_attn_probs),
    )
    if return_attn_probs:
        # real FA returns (out, softmax_lse, S_dmask); vLLM returns (out, softmax_lse)
        o, lse = out
        return o, lse, None
    if isinstance(out, tuple):
        return out[0]
    return out


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    **kwargs,
):
    """Non-varlen flash-attn. vLLM does not expose ``flash_attn_func``; build cu_seqlens
    from the regular ``(batch, seqlen, nheads, headdim)`` shape and delegate to varlen.
    """
    import torch

    assert q.dim() == 4, f"expected (batch, seqlen, nheads, headdim), got {q.shape}"
    batch, seqlen_q, nheads, headdim = q.shape
    seqlen_k = k.shape[1]

    q_flat = q.reshape(batch * seqlen_q, nheads, headdim)
    k_flat = k.reshape(batch * seqlen_k, k.shape[2], headdim)
    v_flat = v.reshape(batch * seqlen_k, v.shape[2], headdim)

    cu_seqlens_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.arange(0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=q.device)

    out = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
    if return_attn_probs:
        o, lse, s = out
        return o.reshape(batch, seqlen_q, nheads, headdim), lse, s
    return out.reshape(batch, seqlen_q, nheads, headdim)


def flash_attn_with_kvcache(*args, **kwargs):
    """vLLM's forward-only build does not expose an equivalent usable here. This name must
    exist because transformers imports it unconditionally in the FA2 branch, but it is only
    used for autoregressive generation (inference), not the training/rmpad path."""
    raise NotImplementedError(
        "flash_attn_with_kvcache is not provided by the vLLM-kernel flash_attn shim."
    )
