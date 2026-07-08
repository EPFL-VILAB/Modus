"""flash_attn_varlen_func with a pure-PyTorch SDPA fallback.

The BAGEL inference path calls ``flash_attn_varlen_func`` (packed variable-length
attention). ``flash-attn`` needs a CUDA-compiled wheel matching the exact
torch/CUDA/python — a hard dependency that is painful on a HF ZeroGPU Space. This
module provides a drop-in replacement that uses ``scaled_dot_product_attention``
when flash-attn is unavailable (or when ``MODUS_FORCE_SDPA_ATTN=1``), so the same
code runs with or without flash-attn.

Correctness note: flash varlen ``causal=True`` uses BOTTOM-RIGHT mask alignment
(query row r attends to keys 0..(Lk-Lq+r)), which differs from SDPA's
``is_causal=True`` (top-left). The KV-cache decode case has Lq < Lk, so we build
the bottom-right mask explicitly instead of using ``is_causal``.
"""
import os
import torch
from torch.nn.functional import scaled_dot_product_attention

try:
    from flash_attn import flash_attn_varlen_func as _flash_varlen
except Exception:  # flash-attn not installed (e.g. HF Space)
    _flash_varlen = None

_FORCE_SDPA = os.environ.get("MODUS_FORCE_SDPA_ATTN", "0") == "1"


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal):
    """q: (Tq, Hq, D); k,v: (Tk, Hkv, D). Per-sequence SDPA over the packed batch."""
    Hq, Hkv = q.shape[1], k.shape[1]
    rep = Hq // Hkv
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    outs = []
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]]                 # (Lq, Hq, D)
        ki = k[ck[i]:ck[i + 1]]                 # (Lk, Hkv, D)
        vi = v[ck[i]:ck[i + 1]]
        Lq, Lk = qi.shape[0], ki.shape[0]
        if rep > 1:                              # GQA: expand kv heads to query heads
            ki = ki.repeat_interleave(rep, dim=1)
            vi = vi.repeat_interleave(rep, dim=1)
        qi = qi.transpose(0, 1).unsqueeze(0)     # (1, H, Lq, D)
        ki = ki.transpose(0, 1).unsqueeze(0)
        vi = vi.transpose(0, 1).unsqueeze(0)
        attn_mask = None
        if causal:
            # bottom-right aligned causal mask (matches flash varlen), Lk >= Lq here.
            qpos = torch.arange(Lq, device=qi.device).unsqueeze(1) + (Lk - Lq)
            kpos = torch.arange(Lk, device=qi.device).unsqueeze(0)
            attn_mask = (kpos <= qpos).unsqueeze(0).unsqueeze(0)  # (1,1,Lq,Lk) bool
        oi = scaled_dot_product_attention(qi, ki, vi, attn_mask=attn_mask)
        outs.append(oi.squeeze(0).transpose(0, 1))               # (Lq, H, D)
    return torch.cat(outs, dim=0)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q=None, max_seqlen_k=None, causal=False, **kwargs):
    if _flash_varlen is not None and not _FORCE_SDPA:
        return _flash_varlen(
            q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, causal=causal,
        )
    return _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
