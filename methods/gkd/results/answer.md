# Generalized Knowledge Distillation (GKD), distilled

GKD is a method for distilling a large autoregressive language teacher into a small student that
fixes two failures of supervised knowledge distillation at once: it trains the student on its
**own on-policy generated sequences** (labeled by the teacher's token-level distribution) to
address the train-inference distribution mismatch, and it lets the objective be **any divergence
on the forward-KL ↔ reverse-KL family (generalized JSD)** so an underpowered student can be
mode-seeking instead of mass-covering. Supervised KD and pure on-policy KD are special cases.

## Problem it solves

Distilling an autoregressive student `p_S^θ` from a queryable teacher `p_T` (full token-level
distributions available). Supervised KD trains on a fixed dataset with forward KL, which (1)
shows the student only fixed prefixes while it conditions on its own prefixes at inference — the
errors compound over the horizon (imitation-learning analysis gives behavioral cloning a
quadratic `O(T²ε)` failure mode, while learner-induced state data supports linear-in-horizon
bounds), and (2) uses the
mass-covering forward KL, so a small student smears probability onto teacher-unlikely tokens and
hallucinates.

## Key idea

Two orthogonal knobs combined into one objective:

- **Data (`λ`)**: the student-data fraction. Train on a mixture of fixed data and the student's
  own on-policy rollouts, the teacher providing token-level feedback on the self-generated
  prefixes. This matches the inference distribution of contexts (the DAgger / interactive-oracle
  fix) and is cheap because the small student does the generating. **The gradient is stopped at
  sampling** — generated sequences are treated as fixed data — so the gradient is supervised-KD-
  like and stable, with no high-variance policy-gradient term.
- **Divergence (`β`)**: the generalized Jensen-Shannon divergence between teacher and student
  token distributions, with mixture `M = β p_T + (1−β) p_S`:

  `D_JSD(β)(p_T ‖ p_S) = β · KL(p_T ‖ M) + (1−β) · KL(p_S ‖ M)`.

  The raw unscaled `D_JSD(β)` goes to zero at the endpoints, but its scaled limits recover the
  KL directions: `lim_{β→0} D_JSD(β)/β = KL(p_T‖p_S)`, and by symmetry the `β→1` end recovers
  `KL(p_S‖p_T)` after scaling by `1−β`. Thus small `β` is forward-KL-like (mass-covering,
  diverse, can hallucinate), while large `β` is reverse-KL-like (mode-seeking, high quality,
  less diverse). JSD is bounded even on disjoint supports, taming early-training KL blow-ups.
  In implementation, exact `β=0` and `β=1` are explicit endpoint branches for forward and
  reverse KL; the interior branch is generalized JSD. The loss function's default `β = 0.5`
  gives symmetric JSD, while experiments can choose other values.

## Objective

```
L_GKD(θ) = (1 − λ) · E_{(x,y)~(X,Y)} [ D(p_T ‖ p_S^θ)(y|x) ]
         +    λ    · E_{x~X} [ E_{y~p_S(·|x)} [ D(p_T ‖ p_S^θ)(y|x) ] ]
```
with stop-gradient on `y ~ p_S`, and the per-sequence divergence length-normalized over tokens:
`D(p_T ‖ p_S^θ)(y|x) = (1/L_y) Σ_n D(p_T(·|y_{<n},x) ‖ p_S^θ(·|y_{<n},x))`.

Special cases: **supervised KD** = (`λ=0`, forward KL); **on-policy KD** = (`λ=1`, forward KL);
mixed-data forward-KL methods = interior `0<λ<1` with forward KL.

## Algorithm

```
Given: teacher p_T, student p_S^θ (already supervised-fine-tuned), data (X, Y),
       student-data fraction λ, divergence D (i.e. β), learning rate η
for step k = 1 … K:
    draw u ~ Uniform(0,1)
    if u ≤ λ:
        sample inputs x from X; generate y ~ p_S^θ(·|x)        # on-policy, stop-grad
    else:
        sample (x, y) from (X, Y)                              # fixed data
    θ ← θ − η · (1/B) Σ_{(x,y)} ∇_θ D(p_T ‖ p_S^θ)(y|x)
```

## RL fine-tuning + on-policy GKD

Because it only needs student samples, GKD composes with reward fine-tuning:
```
E_{x~X} [ (1−α) · E_{y~p_S^θ(·|x)}[ r(y) ]  −  α · E_{y~p_S(·|x)}[ D(p_T ‖ p_S^θ)(y|x) ] ]
```
`α = 1` is pure distillation. This regularizes toward the **teacher** rather than the initial
policy; for minimal disruption to an existing RL workflow, use reverse KL or JSD(0.9).

## Working code

Faithful to the canonical TRL `GKDTrainer` non-fused path: the generalized-JSD loss, the masked
length-normalized reduction, the logit shift, and the training step that replaces the batch with
teacher- or student-generated completions before delegating to the normal trainer update.

```python
import random
import torch
import torch.nn.functional as F
from trl.models.utils import unwrap_model_for_generation


def generalized_jsd_loss(student_logits, teacher_logits, labels=None,
                         beta=0.5, temperature=1.0, reduction="batchmean"):
    """Generalized JSD per-token loss. student_logits, teacher_logits: [B, T, V].
    beta: endpoints branch to KL limits; strict interior uses generalized JSD.
    labels: [B, T], -100 on prompt/padding positions to ignore."""
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs,
                       reduction="none", log_target=True)        # KL(p_T || p_S)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs,
                       reduction="none", log_target=True)        # KL(p_S || p_T)
    else:
        # log M, M = (1-beta)*p_S + beta*p_T, in log space via logsumexp
        beta = torch.tensor(beta, dtype=student_log_probs.dtype)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log(1 - beta),
                         teacher_log_probs + torch.log(beta)]),
            dim=0,
        )
        # F.kl_div(input=log_q, target=log_p, log_target=True) computes KL(p || q)
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs,
                              reduction="none", log_target=True)  # KL(p_T || M)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs,
                              reduction="none", log_target=True)  # KL(p_S || M)
        jsd = beta * kl_teacher + (1 - beta) * kl_student

    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    if reduction == "batchmean":
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    return jsd


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """Student + frozen teacher forward over the same tokens; slice to completion
    positions (logit at n predicts token n+1); take the generalized-JSD loss."""
    student_outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])
    self.teacher_model.eval()
    with torch.no_grad():
        teacher_outputs = self.teacher_model(input_ids=inputs["input_ids"],
                                             attention_mask=inputs["attention_mask"])

    prompt_lengths = inputs["prompts"].shape[1]
    shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
    shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
    shifted_labels = inputs["labels"][:, prompt_lengths:]

    loss = self.generalized_jsd_loss(
        student_logits=shifted_student_logits,
        teacher_logits=shifted_teacher_logits,
        labels=shifted_labels,
        beta=self.beta,
    )
    return (loss, student_outputs) if return_outputs else loss


@staticmethod
def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
    """Roll out the provided generator on the prompts and relabel its generations as the batch.
    The generated token ids are reassigned as fixed training data."""
    generated_outputs = model.generate(
        input_ids=inputs["prompts"],
        attention_mask=inputs.get("prompt_attention_mask", None),
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    generated_tokens = generated_outputs.sequences
    new_attention_mask = torch.ones_like(generated_tokens)
    new_labels = generated_tokens.clone()
    if pad_token_id is not None:
        new_labels[new_labels == pad_token_id] = -100
        new_attention_mask[generated_tokens == pad_token_id] = 0
    return generated_tokens, new_attention_mask, new_labels


def training_step(self, model, inputs, num_items_in_batch=None):
    """With probability lmbda, train on the student's own generations; `seq_kd`
    uses teacher generations. Backward and optimizer handling stay in the parent trainer."""
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

    loss = super().training_step(model, inputs, num_items_in_batch)
    return loss
```

## Relation to prior methods

- **Supervised KD** (Hinton 2015; Sanh 2019) = GKD with `λ=0`, forward KL. Fixed off-policy data,
  mass-covering.
- **SeqKD** (Kim & Rush 2016) = behavioral cloning on the teacher's beam-search mode — fixed,
  teacher-generated, single-mode, off-policy.
- **DAgger** (Ross et al. 2011) = the imitation-learning ancestor that motivates learner-induced
  data: train on the learner's own visited states, labeled by the interactive oracle, replacing the
  behavioral-cloning `O(T²ε)` failure mode with a bound of the form `J(π̂) ≤ J(π*) + O(uTε_N) + O(1)`.
  GKD applies the same state-distribution idea to LMs with a *full-distribution* teacher instead of
  oracle actions.
- **ImitKD** (Lin et al. 2020) = interior `0<λ<1`, forward KL only, token-level — never fully
  on-policy and never beyond forward KL.
- **MiniLLM** (Gu et al. 2023) = sequence-level reverse KL by policy gradient (backprops through
  sampling, needs variance/stability tricks). GKD keeps the supervised-style gradient by
  stop-gradient on sampling, and is general over divergences.
