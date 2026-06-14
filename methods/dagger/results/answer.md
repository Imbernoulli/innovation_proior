# DAgger (Dataset Aggregation), distilled

DAgger is an iterative imitation-learning meta-algorithm that produces a single **stationary,
deterministic** policy whose loss is small under *its own induced state distribution*. Each
round it rolls out the current policy (optionally mixed with the expert), labels every visited
state with the expert's action, **aggregates** those pairs into one growing dataset, and refits
a supervised policy on the whole aggregate. Refitting on all data seen so far is exactly
*Follow-The-Leader*, so DAgger is a reduction of imitation learning to **no-regret online
learning**: any no-regret learner used as the per-round policy-chooser yields a policy with
performance guarantees linear (not quadratic) in the horizon.

## Problem it solves

Imitation learning is sequential, not i.i.d.: the learned policy's own actions determine the
states it later sees, so training on the expert's state distribution `d_{π*}` but deploying on
the policy's own `d_{π̂}` causes compounding errors. The goal is

```
π̂ = argmin_{π ∈ Π} E_{s ∼ d_π}[ ℓ(s,π) ],
```

the surrogate loss under the policy's *own* distribution — a non-convex, chicken-and-egg
objective (`d_π` depends on `π`, dynamics unknown, only samplable by running `π`).

## Why naive behavior cloning fails (the motivating bound)

If `ℓ` upper-bounds the 0-1 loss vs `π*` and `E_{s∼d_{π*}}[ℓ(s,π̂)] = ε`, then
`J(π̂) ≤ J(π*) + T² ε`. The `T²` is **tight** (Kääriäinen 2006; Ross & Bagnell 2010): one early
mistake (prob `~ε`) throws the policy off-distribution, where it has no training signal and pays
cost `~T`. Fix requires changing *which states you train on*, not which classifier you use.

## Key idea

Collect states under the policy's own rollouts, label with the expert, **aggregate and refit**:

```
D ← ∅;  π̂_1 ← any policy in Π
for i = 1 to N:
    π_i = β_i π* + (1 − β_i) π̂_i              # optional per-step expert mixing
    sample T-step trajectories with π_i
    D_i = { (s, π*(s)) : s visited by π_i }    # expert label at learner-visited states
    D ← D ∪ D_i                                # aggregate
    π̂_{i+1} ← train on D                        # Follow-The-Leader (best in hindsight)
return best π̂_i on validation
```

- **β schedule.** `β_1 = 1` (round 1 is pure expert, so no `π̂_1` need be specified). Then
  decay, e.g. `β_i = p^{i-1}`. The only requirement is `\bar β_N = (1/N)Σ_i β_i → 0`, so the
  collected states converge to the *learner's* distribution. Parameter-free default
  `β_i = I(i = 1)` (pure expert round 1, pure learner after) often works best.
- **No free parameters** except the supervised sub-routine; iterations scale ~linearly with the
  effective horizon; handles continuous *and* discrete actions (online-convex, not
  classification-only).

## Guarantees

Treat each round as one online example with loss `ℓ_i(π) = E_{s∼d_{π_i}}[ℓ(s,π)]`; the online
learner chooses the learned policies `π̂_i`, so no-regret is:
`(1/N)Σ ℓ_i(π̂_i) − min_π (1/N)Σ ℓ_i(π) ≤ γ_N → 0` (FTL on strongly convex `ℓ` gives
`γ_N = Õ(1/N)`). Let `ε_N = min_π (1/N) Σ_i E_{s∼d_{π_i}}[ℓ(s,π)]`.

- **Learned policies on collected distributions (infinite sample):** some `π̂_i` has
  `E_{s∼d_{π_i}}[ℓ(s,π̂_i)] ≤ ε_N + γ_N`.
- **Collected → deployed distribution (the key lemma):** `‖d_{π_i} − d_{π̂_i}‖_1 ≤ 2 T β_i`, since
  `d_{π_i} = (1−β_i)^T d_{π̂_i} + (1−(1−β_i)^T) d` and `(1−β)^T ≥ 1 − βT`. So with
  non-increasing `β_i` and `n_β` = largest `i` with `β_i > 1/T`,
  ```
  ∃ π̂ ∈ π̂_{1:N}:  E_{s∼d_{π̂}}[ℓ(s,π̂)] ≤ ε_N + γ_N + (2ℓ_max/N)[ n_β + T Σ_{i=n_β+1}^N β_i ].
  ```
  For `β_i = (1−α)^{i-1}` the bracket is `≤ (1/α)[log T + 1]`, negligible at `N = Õ(T)`.
- **DAgger is itself no-regret:** for nonnegative convex `ℓ_i` with `ℓ_i(π*) ≤ C`, convexity gives
  `ℓ_i(π_i) ≤ β_i C + ℓ_i(π̂_i)`, so DAgger's average regret `≤ γ_N + C \bar β_N → 0`.
- **Task cost:** combining with the forward-training telescoping bound
  (`J(π) ≤ J(π*) + uTε` for cost-to-go gap `u`),
  ```
  J(π̂) ≤ J(π*) + uT(ε_N + γ_N) + O(1)  =  J(π*) + uT ε_N + O(1)   for N = Õ(uT).
  ```
- **Finite sample (`m` trajectories/iter):** the empirical–true gap is a bounded martingale
  (the `π̂_i` are dependent across rounds), so Azuma–Hoeffding gives, w.p. `≥ 1−δ`,
  ```
  E_{s∼d_{π̂}}[ℓ(s,π̂)] ≤ ε̂_N + γ_N
      + (2ℓ_max/N)[n_β + TΣ_{i=n_β+1}^N β_i]
      + ℓ_max √(2 log(1/δ)/(mN)),
  ```
  needing total trajectory count on the order of `T² log(1/δ)` for `O(1/T)` surrogate
  generalization when `ℓ_max` is constant (explicitly, the displayed term is `≤ 1/T` if
  `mN ≥ 2ℓ_max²T² log(1/δ)`), or the `u²T² log(1/δ)` scale when the term is multiplied
  through the `uT` task-cost bound.
  (strong convexity could tighten the surrogate case to `Õ(T log(T/δ))`).

## Forward-training bound (used above), derived

For the policy `π_{1:k}` (run `π` for `k` steps, then `π*`), telescoping:
```
J(π) = J(π*) + Σ_{t=1}^T E_{s∼d^t_π}[ Q^{π*}_{T-t+1}(s,π) − Q^{π*}_{T-t+1}(s,π*) ]
     ≤ J(π*) + u Σ_t E_{s∼d^t_π}[ℓ(s,π)] = J(π*) + uTε,
```
since the bracket is nonzero only when `π ≠ π*` (prob `≤ ℓ(s,π)`), costing `≤ u`. Here `u ≤ 1`
for the 0-1 imitation loss, and `u = O(1)` when `π*` recovers quickly (rapidly-mixing chain).

## Working code — general DAgger loop

```python
import numpy as np


def dagger(expert, env, horizon, n_iters, beta_schedule=None, m=1):
    """Dataset Aggregation: one stationary deterministic policy good under its own
    induced state distribution. Refitting on the aggregate = Follow-The-Leader."""
    if beta_schedule is None:
        beta_schedule = [1.0] + [0.0] * (n_iters - 1)   # beta_i = I(i == 1)

    dataset, policy, candidates = [], None, []
    for i in range(n_iters):
        beta = beta_schedule[i]

        def mixed_act(s, policy=policy, beta=beta):
            # pi_i = beta * expert + (1 - beta) * learner, per step
            if policy is None or np.random.rand() < beta:
                return expert.act(s)
            return policy.act(s)

        for _ in range(m):                              # collect under the mixed policy
            s = env.reset()
            for _ in range(horizon):
                dataset.append((s, expert.act(s)))      # expert label at visited state
                s = env.step(mixed_act(s))

        states = np.stack([s for s, a in dataset])
        actions = np.stack([a for s, a in dataset])
        policy = SupervisedPolicy()                     # refit on ALL data so far (FTL)
        policy.train(states, actions)
        candidates.append(policy)

    return min(candidates, key=lambda p: validation_cost(p, expert, env, horizon))
```

## Working code — token-level DAgger specialization of the GKD trainer path

In an on-policy distillation trainer, the **teacher** is the expert `π*` and the **student** is
the learner. The real GKD trainer path supplies the state mixing: `lmbda` is the learner-side
probability (`1−β_i` in the mixture notation), so with probability `lmbda` it replaces the batch
with generations sampled from the student (learner-induced states), otherwise it keeps the dataset
batch; the non-fused path then runs both models on the chosen batch, shifts
completion logits by `prompt_lengths - 1 : -1`, and masks positions where `labels == -100`.
If sequence-level teacher generation is enabled, that teacher-generated batch is formed before
the `lmbda` student-generation replacement, matching the trainer's order. The stock path computes
generalized JSD on the shifted completion logits; a DAgger specialization keeps the `training_step`
state-mixing path unchanged and replaces `compute_loss` with hard-target cross-entropy onto the
teacher's chosen top-1 token at each completion position:
`ℓ_CE(θ; s, ã) = −Σ_j log π_θ(ã_j | s, ã_{<j})`, with `ignore_index=-100` masking prompt/padding
tokens.

```python
import random
import torch
import torch.nn.functional as F
from transformers.trainer import Trainer
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.gkd_trainer import GKDTrainer
from trl.trainer.utils import empty_cache


class Top1DaggerGKDTrainer(GKDTrainer):
    @staticmethod
    def top1_dagger_loss(student_logits, teacher_logits, labels=None, reduction="batchmean"):
        """Hard expert-action loss: teacher argmax token, averaged over valid completion tokens."""
        target_tokens = teacher_logits.argmax(dim=-1)
        if labels is not None:
            target_tokens = target_tokens.masked_fill(labels == -100, -100)

        flat_targets = target_tokens.reshape(-1)
        flat_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            flat_targets,
            ignore_index=-100,
            reduction="none",
        )

        if labels is not None:
            valid = flat_targets != -100
            flat_loss = flat_loss[valid]
            denom = valid.sum().clamp_min(1).to(flat_loss.dtype)
        else:
            denom = torch.tensor(max(student_logits.size(0), 1), device=flat_loss.device, dtype=flat_loss.dtype)

        if reduction == "batchmean":
            return flat_loss.sum() / denom
        if reduction == "sum":
            return flat_loss.sum()
        if reduction == "mean":
            return flat_loss.mean() if flat_loss.numel() else flat_loss.sum()
        return flat_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Same completion-token alignment as GKDTrainer, with hard teacher actions."""
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]
        loss = self.top1_dagger_loss(
            shifted_student_logits,
            shifted_teacher_logits,
            shifted_labels,
            reduction="batchmean",
        )
        empty_cache()
        return (loss, student_outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Same state mixing as GKDTrainer: seq_kd teacher batch, then lmbda student batch."""
        if self.seq_kd:
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        if random.random() <= self.lmbda:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        return Trainer.training_step(self, model, inputs, num_items_in_batch)
```

The contribution is the **state-distribution fix** (train on the learner's own states, labeled
by the expert) plus the hard expert-action target; soft-KL/JSD distillation losses can use the
same state-mixing hook but a different token-level objective.
