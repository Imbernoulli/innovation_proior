# IPO (Identity Preference Optimisation), distilled

IPO is an offline preference-learning loss that trains a policy `π` directly from
pairwise human preferences `(x, y_w, y_l)` — no reward model, no RL — by
**regressing the reference-corrected log-likelihood-ratio gap of winner over
loser onto a single fixed target `1/(2β)`** (`= τ⁻¹/2`). It is the
`Ψ = identity` special case of a general preference objective; choosing `Ψ`
*bounded* is what keeps KL regularization effective even when preferences are
deterministic, which is exactly where DPO's unbounded sigmoid loss collapses.

## Problem it solves

Given an offline dataset of pairwise preferences and a reference policy `π_ref`,
learn a KL-regularized preference-maximizing policy with a coefficient that truly
controls closeness to `π_ref`. The failure mode to avoid: near-deterministic
preferences (common in finite data over large action/context spaces) making the
regularization silently vanish, so the policy overfits the preferences and
ignores `π_ref` regardless of the coefficient.

## Key idea

Consider the family `J(π) = E_{y~π, y'~μ}[ Ψ(p*(y ≻ y')) ] − τ·KL(π ‖ π_ref)`
over nondecreasing `Ψ: [0,1] → R`. With the Bradley–Terry logit `Ψ(q) =
log(q/(1−q))`, this recovers RLHF and DPO — and that `Ψ` is **unbounded**, so a
deterministic preference forces the implied reward gap to `+∞`, sending the
dispreferred action's probability to 0 for *every* `τ`. Picking a **bounded**
`Ψ` removes the pathology. The simplest bounded monotone choice is the identity,
giving direct regularized maximization of the total preference
`p*(π ≻ μ) − τ·KL(π ‖ π_ref)`, whose optimum is
`π*(y) ∝ π_ref(y) exp(τ⁻¹ p*(y ≻ μ))` with a *bounded* score in the exponent.

## Derivation to an offline loss

1. **Root-finding.** Take ratios of the optimal tilt to cancel the normalizer.
   With `h_π(y,y') = log( π(y)π_ref(y') / (π(y')π_ref(y)) )`, the optimum
   satisfies `h*(y,y') = τ⁻¹(p*(y≻μ) − p*(y'≻μ))`. Fold these per-pair equations
   into one objective:

   `L(π) = E_{y,y'~μ}[ ( h_π(y,y') − (p*(y≻μ) − p*(y'≻μ))/τ )² ].`

2. **Uniqueness.** Parametrize `π` by logits `s`; `L` is a positive-semidefinite
   quadratic `∝ Σ μ(y)μ(y')(s(y)−s(y'))²` plus linear/constant terms, hence
   convex, strictly so except along the all-ones logit direction — which leaves
   the softmax policy unchanged. So `π*` is the unique local/global minimizing
   policy, provided `Supp(μ) = Supp(π_ref)` (otherwise underconstrained pairs
   give infinitely many minima).

3. **Sampled loss.** `p*` is unobservable; only Bernoulli labels `I(y,y')` are.
   The sampled loss `E_{y,y'~μ}[ ( h_π(y,y') − τ⁻¹ I(y,y') )² ]` equals `L(π)` up
   to a `π`-independent constant. Proof matches the cross-terms: using
   additivity/antisymmetry of `h_π`, iid `y,y'`, `E_μ[p*(y≻μ)]=1/2`,
   `E_{y'}[I|y]=p*(y≻μ)`, `E_y[I|y']=1−p*(y'≻μ)`, both sides equal
   `E_μ[(2p*(y≻μ)−1)(log π(y) − log π_ref(y))]`.

4. **Empirical loss.** Each recorded comparison `(y_w,y_l)` contributes both
   `(y,y',I)=(y_w,y_l,1)` and `(y_l,y_w,0)`; averaging and using
   `h_π(y_l,y_w) = −h_π(y_w,y_l)`, then completing the square,

   `2H² − 2τ⁻¹H + τ⁻² = 2(H − τ⁻¹/2)² + τ⁻²/2`, with `H = h_π(y_w,y_l)`, so
   `(1/2)[(H−τ⁻¹)² + H²] = (H − τ⁻¹/2)² + τ⁻²/4`,

   so the loss reduces to a single squared term.

## Final algorithm

Starting from `π = π_ref`, minimize

```
L_IPO(π) = E_{(x, y_w, y_l) ~ D} [ ( h_π(y_w, y_l, x) − 1/(2β) )² ],
h_π(y_w, y_l, x) = log( π(y_w|x) π_ref(y_l|x) / ( π(y_l|x) π_ref(y_w|x) ) )
              = [log π(y_w|x) − log π_ref(y_w|x)] − [log π(y_l|x) − log π_ref(y_l|x)],
```

where `β` is the KL-regularization coefficient (`τ` in the derivation; target
`1/(2β) = τ⁻¹/2`). Smaller `β` (weaker regularization) ⇒ larger allowed gap;
larger `β` ⇒ policy pulled toward `π_ref`'s own separation. On the deterministic
two-action instance (`p*(y_1≻y_2)=1`, uniform `π_ref, μ`), `p*(y_1≻μ)=3/4`,
`p*(y_2≻μ)=1/4`, so `π*(y_1)=σ(τ⁻¹/2)`: `τ→∞` gives uniform `π_ref`, `τ→0` gives
the deterministic policy — the coefficient sweeps cleanly, which the logit
objective could not.

## Relation to prior methods

- **DPO** (Rafailov et al. 2023) = the `Ψ = logit` member: `−log σ(τ·h_π)`. Its
  unbounded sigmoid loss has no finite resting point on deterministic
  preferences, so KL regularization vanishes. IPO replaces the saturating
  sigmoid with a squared distance to a finite target.
- **RLHF + PPO** = also `Ψ = logit`, optimized via an explicit reward model and
  RL; partly shielded from the collapse by accidental reward underfitting, at the
  cost of a multi-stage, on-policy pipeline.

## Working code

Filling the `preference_objective` slot of the offline-preference harness; in the
sequence setting the four log-probs are **averaged per completion token** so the
single fixed target is comparable across lengths.

```python
import torch


def ipo_loss(policy_chosen_lp, policy_rejected_lp,
             ref_chosen_lp, ref_rejected_lp,
             chosen_len, rejected_len, beta=0.1):
    """IPO loss for a batch of preference pairs.

    Sequence-summed log-probs of policy/reference on chosen/rejected, the two
    completion lengths, and beta (= KL coefficient tau; target = 1/(2*beta)).
    """
    chosen_len = chosen_len.clamp_min(1).to(policy_chosen_lp.dtype)
    rejected_len = rejected_len.clamp_min(1).to(policy_rejected_lp.dtype)

    # per-token average log-probs (length normalization)
    pol_c = policy_chosen_lp   / chosen_len
    pol_r = policy_rejected_lp / rejected_len
    ref_c = ref_chosen_lp      / chosen_len
    ref_r = ref_rejected_lp    / rejected_len

    # h_pi = [log pi(y_w) - log pi_ref(y_w)] - [log pi(y_l) - log pi_ref(y_l)]
    chosen_logratio   = pol_c - ref_c
    rejected_logratio = pol_r - ref_r
    h_pi = chosen_logratio - rejected_logratio

    # regress the gap onto the single fixed target 1/(2*beta) = tau^{-1}/2
    losses = (h_pi - 1.0 / (2.0 * beta)) ** 2
    return losses.mean()
```

Full trainer form (frozen reference, length-averaged sequence log-probs, the
same `h_pi` and squared target, with implicit-reward logging signals
`beta * (log pi - log pi_ref)`). This mirrors the implementation pattern where
the log-probs are averaged upstream; the equivalent TRL branch divides the
chosen and rejected reference-corrected scores by completion length inside the
`loss_type="ipo"` branch before applying the same squared target.

```python
import torch


class IPOTrainer:
    def __init__(self, policy_model, reference_model, beta=0.1, lr=5e-7):
        self.policy = policy_model
        self.ref = reference_model.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.beta = beta
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    @staticmethod
    def avg_seq_logp(model, input_ids, completion_mask):
        logits = model(input_ids).logits[:, :-1, :].log_softmax(-1)
        labels = input_ids[:, 1:]
        per_token = torch.gather(logits, 2, labels.unsqueeze(2)).squeeze(2)
        mask = completion_mask[:, 1:]
        length = mask.sum(-1).clamp_min(1).to(per_token.dtype)
        return (per_token * mask).sum(-1) / length          # average log-prob

    def compute_preference_loss(self, pol_c, pol_r, ref_c, ref_r):
        h_pi = (pol_c - ref_c) - (pol_r - ref_r)            # reference-corrected gap
        losses = (h_pi - 1.0 / (2.0 * self.beta)) ** 2      # IPO: squared distance to fixed target
        chosen_rewards = self.beta * (pol_c - ref_c).detach()
        rejected_rewards = self.beta * (pol_r - ref_r).detach()
        return losses.mean(), chosen_rewards, rejected_rewards

    def train_step(self, batch):
        pol_c = self.avg_seq_logp(self.policy, batch["chosen_ids"],   batch["chosen_mask"])
        pol_r = self.avg_seq_logp(self.policy, batch["rejected_ids"], batch["rejected_mask"])
        with torch.no_grad():
            ref_c = self.avg_seq_logp(self.ref, batch["chosen_ids"],   batch["chosen_mask"])
            ref_r = self.avg_seq_logp(self.ref, batch["rejected_ids"], batch["rejected_mask"])
        loss, _, _ = self.compute_preference_loss(pol_c, pol_r, ref_c, ref_r)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss
```
