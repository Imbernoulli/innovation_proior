# Context: Learning a Policy Directly from Pairwise Human Preferences

## Research question

We have a generative policy — a language model, or in the cleanest abstraction a
contextual-bandit policy `π(y|x)` that picks an action (a generation) `y` for a
context (prompt) `x` — and a fixed, offline dataset of *pairwise* human
judgments: for many contexts, two candidate generations were shown to a rater
who said which one they preferred, `y_w ≻ y_l`. We want to turn that dataset
into a better policy: one that produces the generations humans prefer, while
staying close to a known reference policy `π_ref` (the base / supervised
model), so that it does not drift, collapse onto a few high-scoring outputs, or
lose fluency. The closeness-to-reference constraint is usually expressed as a KL
penalty with a coefficient that is supposed to control *how far* the learned
policy may move from `π_ref`.

The precise problem: design an *offline* training objective — a loss computed
directly on the recorded `(x, y_w, y_l)` pairs, with no reward model and no
online RL rollouts — whose optimum is the KL-regularized preference-maximizing
policy, and whose regularization coefficient actually does what it claims:
genuinely keep the solution near `π_ref` by a controllable amount. The hard part
is that the preference signal in real data is frequently *near-deterministic* —
for a given pair one generation almost always wins, and in the finite-sample
regime the empirical preference for a pair is often exactly 0 or 1. An objective
that behaves well on soft preferences but lets the regularization quietly
evaporate on hard, deterministic ones is not good enough, because for
large-vocabulary, large-context models the deterministic / sparsely-observed
regime is the common case.

## Background

**Learning from human preferences as a regularized bandit problem.** Aligning
pretrained and instruction-tuned generative models to human desiderata
(Christiano et al. 2017; Stiennon et al. 2020; Ouyang et al. 2022) is framed as
an offline contextual bandit: given context `x`, choose an action `y` that a
human rater most prefers, subject to staying close to a reference policy
`π_ref`. Closeness is enforced by KL regularization (Geist et al. 2019), whose
role is to avoid model drift (Lazaridou et al. 2020; Lu et al. 2020).

**The Bradley–Terry model and pointwise rewards** (Bradley & Terry 1952). The
dominant way to model preferences posits a latent scalar reward (an Elo score)
`r(x,y)` and writes the probability that `y` beats `y'` as a sigmoid of the
reward difference,

  p(y ≻ y' | x) = σ( r(x,y) − r(x,y') ).

Fitting `r` by logistic regression on the preference data, then maximizing it,
is the standard route. The model depends only on reward *differences*, so `r` is
identifiable only up to an additive function of `x`.

**The KL-regularized objective and its closed-form optimum.** The prevailing RL
pipeline (Ziegler et al. 2019; Stiennon et al. 2020; Bai et al. 2022; Ouyang et
al. 2022) maximizes

  J(π) = E_{x~ρ, y~π(·|x)}[ r(x,y) ] − τ · KL( π(·|x) ‖ π_ref(·|x) ),

with `τ > 0` the regularization coefficient. A standard result in the
reward-weighted / advantage-weighted regression line (Peters & Schaal 2007; Peng
et al. 2019) and the control-as-inference line (Levine 2018; Korbak et al. 2022)
is that the maximizer of a KL-regularized reward objective is an exponential
tilt of the reference,

  π*(y|x) ∝ π_ref(y|x) · exp( τ⁻¹ r(x,y) ).

This is exact and worth holding onto: for *any* per-action scalar score `f(y)`,
the maximizer of `E_π[f] − τ KL(π‖π_ref)` is `π*(y) ∝ π_ref(y) exp(τ⁻¹ f(y))`.
The proof is the `−KL(π‖π*) + const` identity (the objective, divided by `τ`,
equals minus the KL from `π` to that softmax target plus a `π`-independent
log-normalizer), so the regularized argmax is unique whenever `π_ref` has full
support.

**Two approximations baked into the standard pipeline.** The reward-based route
rests on two assumptions. First, that pairwise preferences can be *substituted*
by pointwise rewards through the Bradley–Terry sigmoid — i.e. that a single Elo
score per action explains the comparisons. Second, that a reward model fit on
the collected data generalizes to the out-of-distribution generations the policy
will later sample. Direct, reward-model-free methods (below) remove the second
approximation but still lean entirely on the first.

**Diagnostic facts about deterministic preferences (knowable before any new
method).** Two observations about the Bradley–Terry / pointwise-reward route are
load-bearing here, because they are facts about the *prior art*, not about
anything new:

- *Deterministic preferences push Elo scores to infinity.* If a pair has
  `p*(y ≻ y') = 1`, the Bradley–Terry model can only represent it by sending the
  reward gap `r(y) − r(y') → +∞`. Plug an unbounded reward gap into the
  closed-form optimal policy `π*(y) ∝ π_ref(y) exp(τ⁻¹ r(y))` and the
  dispreferred action is driven to probability zero, `π*(y') = 0`, *no matter
  how large `τ` is*. The strength of the KL term, as the preferences approach
  determinism, effectively goes to zero.
- *Finite data manufactures determinism.* Even when the true preference is soft,
  e.g. `p*(y ≻ y') = 0.8`, a handful of samples can give an empirical estimate of
  exactly `1`, which then triggers the same collapse. For large action and
  context spaces — language models — many pairs are observed once or never, so
  empirical preferences land in `{0, 1}` constantly.
- *Why the reward-model route is, in practice, partly shielded.* When empirical
  preferences are in `{0,1}` the Bradley–Terry reward MLE has infinite optima
  that practical training never reaches; the reward function ends up *underfit*,
  and reward regularization is a known, deliberate part of RL-from-feedback
  training (Christiano et al. 2017). That underfitting is what keeps the eventual
  policy regularized toward `π_ref`. A method that fits no explicit reward forgoes
  this incidental shielding.

## Baselines

These are the methods a new offline preference objective would be compared
against and would react to. For each: its mechanism, its math, and the specific
place it stalls.

**Three-stage RLHF with PPO** (Stiennon et al. 2020; Ouyang et al. 2022; Bai et
al. 2022). Stage 1: supervised fine-tune to `π_SFT`. Stage 2: fit a
Bradley–Terry reward model `r_φ` by minimizing `−E[log σ(r_φ(x,y_w) −
r_φ(x,y_l))]`. Stage 3: maximize the KL-regularized objective `J(π)` above with
PPO (Schulman et al. 2017), against the learned `r_φ`. **Gap:** three coupled
models, on-policy sampling of a multi-billion-parameter LM inside the loop, a
value baseline to learn, and sensitivity to reward normalization and to `τ` —
complex and unstable to run at scale. It also rests on the first approximation
(pairwise preferences ↦ pointwise reward).

**Direct Preference Optimisation (DPO)** (Rafailov et al. 2023). Removes the
reward model and the RL loop. Its move is a change of variables: substitute the
closed-form optimum `π*(y) ∝ π_ref(y) exp(τ⁻¹ r(y))` back into the Bradley–Terry
likelihood, which expresses the implicit reward as `r(x,y) = τ log(π(y|x) /
π_ref(y|x)) + τ log Z(x)`; because Bradley–Terry depends only on reward
*differences*, the intractable normalizer `Z(x)` cancels, leaving a supervised
loss over `π` alone,

  L_DPO(π) = −E_{(x,y_w,y_l)~D} [ log σ( τ log(π(y_w|x)/π_ref(y_w|x))
                                       − τ log(π(y_l|x)/π_ref(y_l|x)) ) ].

Its gradient up-weights pairs the implicit reward currently mis-ranks, scaled by
`σ` of the mis-ranking. Simpler, stable, competitive with RLHF. **Gap:** it
still relies on the first approximation — the Bradley–Terry substitution lives
inside the `log σ`, whose argument is an *unbounded* logit. So DPO inherits the
deterministic-preference pathology in full: where the empirical preference for a
pair is `1`, the loss keeps rewarding ever-larger log-ratios `log(π(y_w)/π(y_l))`
with no finite resting point, the term that should hold the policy near `π_ref`
stops binding, and — having fit no explicit reward — DPO lacks even the
incidental underfitting shield that the RLHF reward model provides. The failure
mode to watch is a policy update whose sensitivity to `τ` disappears precisely
when empirical labels become deterministic.

**SLiC-HF** (Zhao et al. 2023). A reward-model-free calibration/contrastive loss
on sequence likelihoods that, like DPO, optimizes the policy directly from
preferences with a margin-style objective. **Gap:** same family of concerns — a
contrastive term on log-likelihoods whose strength, relative to the
reference-anchoring term, has no closed-form guarantee that it will keep the
policy near `π_ref` in the deterministic regime.

**Reward-weighted / advantage-weighted regression** (Peters & Schaal 2007; Peng
et al. 2019). Turn the KL-regularized problem into supervised regression by
weighting sampled actions by `exp(reward/τ)`. **Gap:** still needs reward values
and sampling; it approximates the same closed-form optimum rather than solving
the offline preference problem directly.

**Preference-based / dueling-bandit theory** (Busa-Fekete et al. 2013;
Novoseller et al. 2020; Pacchiano et al. 2023; Wang et al. 2023). Provides regret
guarantees for learning from comparisons, and substitutes a *von Neumann winner*
for an optimal arm when no absolute reward exists. **Gap:** these analyses live
in the standard online bandit setting with regret bounds; they do not address
the offline, KL-regularized, large-model practice that RLHF and DPO operate in.

## Evaluation settings

The natural yardsticks for an offline preference method at this time, settings
only:

- **Synthetic bandit instances with known preferences.** A small discrete action
  set (e.g. two or three actions `{y_a, y_b, y_c}`) with a hand-specified true
  preference function `p*`, uniform `π_ref` and behavior policy `μ`, and a
  dataset of sampled comparisons. Policies are encoded as a softmax over a logit
  vector `θ ∈ R^|Y|`. This setting is chosen precisely because the closed-form
  optimal policy is computable, so one can read off how close the learned policy
  is to `π_ref` as the regularization coefficient is swept.
- **Deterministic-preference instance.** Two actions with `p*(y_1 ≻ y_2) = 1`,
  uniform `π_ref`, sweeping the regularization coefficient `τ` — the minimal
  setting that exposes whether regularization survives a deterministic preference.
- **Total-ordering vs. cyclic small datasets.** Three actions, datasets such as
  `D_1 = {(y_a,y_b),(y_b,y_c),(y_a,y_c)}` (a total order) and `D_2 =
  {(y_a,y_b),(y_b,y_c),(y_c,y_a)}` (a cycle); and a dataset `D_3 =
  {(y_a,y_b),(y_b,y_a)}` that leaves a pair unobserved — to probe behavior when
  one action dominates, and when an action is never observed winning.
- **Optimization protocol.** Softmax-parametrized policies trained with a
  first-order optimizer (Adam, lr 0.01) for a fixed number of steps, minibatches
  drawn by uniform sampling with replacement; each configuration repeated over
  several seeds, reporting mean and confidence intervals. Implemented in
  JAX/Flax with the optimizer from Optax.
- **Language-model setting.** A causal LM policy and a frozen reference of the
  same architecture, a preference dataset of `(prompt, chosen, rejected)`
  triples, sequence log-probabilities formed over the completion tokens, and a
  scalar regularization coefficient — the same harness any offline preference
  loss plugs into.

## Code framework

The primitives that already exist: a causal-LM policy `π_θ` and a frozen
reference `π_ref` of the same architecture, a tokenizer, a first-order optimizer,
and a preference dataset yielding, per example, a prompt with a *chosen* and a
*rejected* completion. From these we can form, per example, four
sequence-level log-probabilities — the policy and the reference, each on the
chosen and the rejected completion. What is missing is the training objective
that maps a preference pair to a scalar loss on the policy.

```python
import torch

# --- given primitives ---
policy_model    = load_causal_lm()           # trainable π_θ, returns logits
reference_model = load_causal_lm().eval()    # frozen anchor π_ref
for p in reference_model.parameters():
    p.requires_grad_(False)
optimizer = torch.optim.Adam(policy_model.parameters(), lr=5e-7)


def sequence_logprob(model, input_ids, completion_mask):
    """Per-token log p(token) summed over the completion positions, and the
    completion length, for each example in the batch."""
    logits = model(input_ids).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    per_token = torch.gather(logits.log_softmax(-1), 2,
                             labels.unsqueeze(2)).squeeze(2)
    mask = completion_mask[:, 1:]
    length = mask.sum(-1).clamp_min(1).to(per_token.dtype)
    return (per_token * mask).sum(-1), length           # (logp_sum, length), shape (B,)


def preference_objective(policy_chosen_lp, policy_rejected_lp,
                         ref_chosen_lp, ref_rejected_lp,
                         chosen_len, rejected_len, beta):
    """Map a preference pair to a scalar loss on π_θ.

    Inputs: the four sequence log-probs (policy and reference, on the chosen and
    rejected completions), the two completion lengths, and the regularization
    coefficient beta. Returns a scalar loss.
    """
    # TODO: the offline preference objective we will design
    pass


def train_step(batch, beta):
    pc, lc = sequence_logprob(policy_model,    batch["chosen_ids"],   batch["chosen_mask"])
    pr, lr = sequence_logprob(policy_model,    batch["rejected_ids"], batch["rejected_mask"])
    with torch.no_grad():  # reference is frozen
        rc, _ = sequence_logprob(reference_model, batch["chosen_ids"],   batch["chosen_mask"])
        rr, _ = sequence_logprob(reference_model, batch["rejected_ids"], batch["rejected_mask"])
    loss = preference_objective(pc, pr, rc, rr, lc, lr, beta)   # TODO returns scalar
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss
```

The single empty slot is `preference_objective`: the map from a preference pair
to a scalar loss on the policy. Everything around it — the LM forward pass, the
masked sequence log-probability, the frozen reference, the optimizer loop — is
standard and already on the table.
