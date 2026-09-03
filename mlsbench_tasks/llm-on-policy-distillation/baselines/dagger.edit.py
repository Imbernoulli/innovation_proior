"""DAgger baseline — Revisiting DAgger in the Era of LLM-Agents
(arXiv 2605.12913).

Reference loss (Eq. 12-13, Section 3.2):

    L(θ) = E_{(s_t, ã_t) ~ D} [ Σ_j ℓ_CE(θ; s_t, ã_t) ]
    ℓ_CE(θ; s_t, ã_t) = − Σ_j log π_θ(ã_{t,j} | s_t, ã_{t,<j})

i.e. plain cross-entropy on the *teacher's chosen action* at each state, with
no KL term, no temperature, no advantage weighting (Section 3.3 Table 2: state
distribution is the mixed-policy DAgger rollout, label source is π_teacher,
weight w(s,a) = 1).

In a single-turn distillation setting the per-token teacher action is the
teacher's top-1 (argmax) over its conditional distribution at the same state
the student is forwarding. This is the loss the paper reports for DAgger; its
empirical claim (Table 3) is +3.9/+3.6 pp over Vanilla-OPD's reverse-KL on
SWE-Bench Verified.

For math reasoning at the 0.5B-student / 7B-Math-teacher scale, whether this
loss beats reverse-KL (OPD) or generalized-JSD (GKD) is open — this baseline
provides the comparison.
"""

_FILE = "trl/trl/experimental/gkd/custom_distill_loss.py"

_BODY = """\
    # DAgger (arXiv 2605.12913) — cross-entropy on teacher's top-1 action.
    # Eq.12-13 + §A: plain CE with unit token weight (w(s,a)=1), no temperature
    # rescaling on either side. The state distribution comes from upstream
    # GKDTrainer's lmbda mixing (with prob lmbda, inputs are student-generated;
    # otherwise dataset). This loss replaces OPD's reverse-KL soft target with
    # a hard-target supervision signal at each token position.

    # No temperature scaling: paper specifies plain log π_θ(ã|s).
    # Teacher's argmax action at each position is the deterministic
    # instantiation of the teacher's chosen action at that state.
    target_tokens = teacher_logits.argmax(dim=-1)  # [B, T]

    if labels is not None:
        # Preserve the -100 padding mask from `labels`.
        target_tokens = torch.where(labels == -100, labels, target_tokens)

    # Token-level CE.
    per_token = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        target_tokens.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )  # [B*T]

    if labels is not None:
        valid_mask = (labels != -100).reshape(-1)
        denom = valid_mask.sum().clamp_min(1)
        loss_sum = per_token.sum()
    else:
        denom = torch.tensor(max(per_token.numel(), 1), device=per_token.device, dtype=per_token.dtype)
        loss_sum = per_token.sum()

    if reduction == "batchmean":
        return loss_sum / denom
    elif reduction == "sum":
        return loss_sum
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
