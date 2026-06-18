# Context: Preventing Preference Training from Lowering Preferred Likelihood

## Research question

Direct preference training fits a policy from triples `(x, y_w, y_l)`, where `y_w`
is preferred to `y_l`, without training a separate reward model or running an RL
loop. The attractive part is that the loss is just a supervised binary
cross-entropy on a reference-relative log-ratio gap. The uncomfortable part is
that the loss only asks for the preferred completion to beat the dispreferred one
relatively. It does not separately ask whether the preferred completion is still
likely under the new policy.

That distinction matters when the preferred completion is not merely "better than
the other sample" but actually a satisfactory target: distilled reasoning traces,
responses converted from labeled datasets, or preference data where the positive
side is intended to remain a good completion. The central question is whether a
policy can reduce the likelihood of `y_w` while still improving the preference
loss, and if so, what minimal change to the objective supplies restoring pressure
on `y_w` without giving up the direct preference-learning setup.

## Background

**KL-constrained reward optimization.** RLHF usually starts from
`max_pi E_{y~pi}[r(x,y)] - beta KL(pi || pi_ref)`. The solution has the
exponential-tilt form
`pi_r(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x)`, so
`r(x,y) = beta log(pi_r(y|x)/pi_ref(y|x)) + beta log Z(x)`.
In pairwise Bradley-Terry preferences, the `beta log Z(x)` term cancels because
it depends only on the prompt.

**Direct preference optimization.** With
`rho(y) = log pi_theta(y|x) - log pi_ref(y|x)`, the direct loss is

```text
L = -E log sigma(beta * (rho(y_w) - rho(y_l))).
```

Equivalently, the implicit reward is
`r_hat(x,y) = beta log(pi_theta(y|x)/pi_ref(y|x))`. The update raises the
preferred log-probability and lowers the dispreferred log-probability with a
self-paced weight `sigma(r_hat(y_l) - r_hat(y_w))`, largest when the pair is
misordered.

**Why the gap alone can be insufficient.** The scalar
`rho(y_w) - rho(y_l)` can increase either because `rho(y_w)` rises or because
`rho(y_l)` falls. The loss does not distinguish these routes. When the two
completions differ by only a few tokens, especially in chain-of-thought or
calculation data, suppressing the dispreferred continuation can also suppress
shared continuation tokens used by the preferred one.

**The one-token diagnostic.** For two completions that first differ at token
position `m`, all earlier positions cancel out of the contrastive gradient. For
later positions, the token-level logit update direction is controlled by
differences of the model's next-token probabilities under the preferred and
dispreferred prefixes. Under the usual assumption that the starting model already
assigns higher probability to the correct next token under the preferred prefix,
the contrastive update can move the later correct-token logits in the wrong
direction. This is the local mechanism behind preferred likelihood erosion on
low-edit-distance pairs.

## Baselines

**DPO.** The direct Bradley-Terry loss on the reference-relative log-ratio gap.
It is the base objective and the main point of comparison. Its gap-only nature is
the failure route under examination.

**Preferred-only supervised fine-tuning.** Maximizes likelihood on the preferred
completion. It directly protects positive likelihood, but it discards the
negative completion and loses the pairwise contrast that makes preference data
useful.

**SLiC-HF.** Uses a margin ranking loss
`max(0, delta - log pi(y_w|x) + log pi(y_l|x))` plus a cross-entropy
regularizer. It is a direct, pairwise, non-RL baseline with an explicit margin,
but it does not use the same reference-relative Bradley-Terry parameterization.

**IPO.** Replaces the saturating logistic objective with squared regression of
the reference-corrected log-ratio gap toward a fixed target `1/(2 beta)` (or
`tau^{-1}/2` in its notation). It controls overfitting of the gap, but it does
not directly impose a one-sided penalty on preferred likelihood falling below the
reference.

## Evaluation settings

The low-edit-distance setting is a paired version of MetaMath/GSM8K: the
preferred completion is a correct reasoning trace, and the dispreferred
completion is produced by corrupting an intermediate calculation while leaving
most text unchanged. Its normalized edit distance is about `6.5%`, so it stresses
the shared-token failure mode.

The high-edit-distance setting is ARC-Challenge converted to preference pairs by
pairing the correct answer with each incorrect choice; its normalized edit
distance is about `90%`. HellaSwag is converted similarly for larger-model
training. The core comparison trains DPO, IPO, SLiC-HF, and the modified direct
preference loss on Mistral-7B and evaluates with the LLM Evaluation Harness on
GSM8K and ARC.

The training diagnostics to watch are the preferred-completion log-probability
during training and token-level log-probabilities around the first differing
token. The large-model setting mixes six preference datasets, trains 7B, 34B,
and 72B models with the same direct-preference infrastructure, and evaluates
generalization with MT-Bench and the HuggingFace Open LLM Leaderboard suite.

## Code framework

The available training primitive is a DPO-style trainer. For each batch it
computes summed completion log-probabilities under the policy and a frozen
reference model:

```python
import torch
import torch.nn.functional as F


def sequence_logprob(model, input_ids, loss_mask):
    """Summed per-token log p(token) over masked completion positions."""
    logits = model(input_ids).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    per_token = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(2)).squeeze(2)
    return (per_token * loss_mask[:, 1:]).sum(-1)


def preference_objective(policy_chosen_lp, policy_rejected_lp,
                         ref_chosen_lp, ref_rejected_lp, beta):
    """Map the four summed sequence log-probs to a per-example loss.

    Standard DPO uses only the difference of policy and reference log-ratio
    gaps. The missing design choice is how to add preferred-likelihood
    protection while keeping the same reference-based preference objective.
    """
    # TODO: define the modified direct preference loss
    pass
```
