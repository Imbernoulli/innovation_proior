"""RS-KD baseline — Random Sampling KD with importance-corrected top-K
(Anshumann et al., ACL'25 oral, arXiv 2503.16870).

Reference impl: https://github.com/akhilkedia/RandomSamplingKD

Insight: naive top-K caching of teacher logits is *biased* because it
drops out-of-cache mass. RS-KD restores unbiasedness by:

    1. Picking the K tokens with highest teacher prob per position.
    2. Adding a tail "(K+1)-th" bucket that accumulates the remaining mass,
       so the (K+1)-element distribution is a proper probability vector.
    3. Computing KL over the K+1 buckets on the *same* support for student
       and teacher.

The student tail bucket is constructed analogously: log(1 - Σ_{v∈topK} p_S(v)).
This formulation matches the `_add_tail_bucket` trick used in TRL's
experimental DistillationTrainer.

We pin K = 128 (close to the open-source RS-KD default; small enough to keep
memory tame on a 0.5B/7B pair with Qwen vocab size 151936).
"""

_FILE = "trl/trl/experimental/gkd/custom_distill_loss.py"

_BODY = """\
    # RS-KD (Anshumann et al., ACL'25) — sparse top-K KL with explicit tail bucket.
    top_k = 128
    eps = 1e-9
    log_one_minus_eps = -1e-7  # ensures log(1 - sum(exp(top_k))) is finite

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Select teacher's top-K indices per position; gather student log-probs on the same support.
    K = min(int(top_k), teacher_log_probs.size(-1))
    teacher_topk_logp, topk_idx = torch.topk(teacher_log_probs, k=K, dim=-1)  # [B, T, K]
    student_topk_logp = torch.gather(student_log_probs, dim=-1, index=topk_idx)

    # Tail buckets: log(1 - sum(exp(top_k_logp))). Clamp the inner sum < 1 for numerical safety.
    def _tail(logp_topk):
        log_sum = torch.logsumexp(logp_topk, dim=-1, keepdim=True).clamp(max=log_one_minus_eps)
        # log(1 - exp(log_sum))  via log(-expm1(log_sum))
        return torch.log(-torch.expm1(log_sum))

    teacher_full = torch.cat([teacher_topk_logp, _tail(teacher_topk_logp)], dim=-1)  # [B, T, K+1]
    student_full = torch.cat([student_topk_logp, _tail(student_topk_logp)], dim=-1)

    # Renormalise (the topk + tail decomposition is exact in expectation but
    # numerical errors can push it slightly off; a small renorm keeps the
    # divergence well-defined).
    teacher_full = teacher_full - torch.logsumexp(teacher_full, dim=-1, keepdim=True)
    student_full = student_full - torch.logsumexp(student_full, dim=-1, keepdim=True)

    # KL(p_T || p_S) over the K+1 support.
    per_token = F.kl_div(
        student_full, teacher_full, reduction="none", log_target=True
    ).sum(dim=-1)  # [B, T]

    if labels is not None:
        mask = labels != -100
        per_token = per_token[mask]
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(max(per_token.numel(), 1), device=per_token.device, dtype=per_token.dtype)

    if reduction == "batchmean":
        return per_token.sum() / denom
    elif reduction == "sum":
        return per_token.sum()
    elif reduction == "mean":
        return per_token.mean()
    return per_token
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 42,
        "end_line": 65,
        "content": _BODY,
    },
]
