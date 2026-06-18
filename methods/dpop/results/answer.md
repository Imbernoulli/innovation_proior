# DPO-Positive (DPOP)

DPOP is DPO with a soft, one-sided preferred-likelihood penalty. For
`rho(y) = log pi_theta(y|x) - log pi_ref(y|x)`, standard DPO uses only the gap
`rho(y_w) - rho(y_l)`. DPOP changes the pre-sigmoid score to

```text
rho(y_w) - rho(y_l) - lambda * max(0, -rho(y_w))
```

so the loss is

```text
L_DPOP = -E log sigma(beta * (
    log(pi_theta(y_w|x) / pi_ref(y_w|x))
  - log(pi_theta(y_l|x) / pi_ref(y_l|x))
  - lambda * max(0, log(pi_ref(y_w|x) / pi_theta(y_w|x)))
)).
```

`lambda = 0` recovers DPO exactly. If `rho(y_w) >= 0`, the hinge is zero and the
example has the ordinary DPO loss. If `rho(y_w) < 0`, the logit becomes
`(1 + lambda) * rho(y_w) - rho(y_l)`, adding preferred-likelihood restoring
pressure while retaining the rejected-completion contrast.

For the active hinge case, the token-level update direction after a one-token
edit has these two cases:

```text
correct token i:  lambda * (1 - s_i^w) + s_i^l - s_i^w
other token j:   -(lambda + 1) * s_j^w + s_j^l
```

For large enough `lambda`, the correct-token direction is positive and the
non-token directions are negative, reversing the wrong-way continuation update
that DPO can produce after the first differing token.

The implementation is the standard TRL-style DPO score with one line added before
`logsigmoid`:

```python
import torch
import torch.nn.functional as F


def dpop_loss(policy_chosen_logps, policy_rejected_logps,
              reference_chosen_logps, reference_rejected_logps,
              beta, lambda_dpop, label_smoothing=0.0):
    """DPOP on summed completion log-probs."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    penalty = torch.clamp(reference_chosen_logps - policy_chosen_logps, min=0.0)
    logits = logits - lambda_dpop * penalty

    losses = (
        -F.logsigmoid(beta * logits) * (1.0 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards
```

Use summed, not length-averaged, completion log-probs for the DPO/DPOP
Bradley-Terry form. The default experimental recipe uses `beta = 0.3`,
`lambda = 50`, AdamW, learning rate `5e-5`, and 1000 DPO/DPOP steps; ablations
cover `beta in {0.1, 0.3, 1.0}` and `lambda in {5, 50, 500}`.

Reference: Pal et al., *Smaug: Fixing Failure Modes of Preference Optimisation
with DPO-Positive*, arXiv:2402.13228v2.
