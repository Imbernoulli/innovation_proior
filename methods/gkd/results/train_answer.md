Compressing a large autoregressive language model into a much smaller student is harder than ordinary classification distillation because the student does not just predict labels for a fixed set of inputs; it generates sequences one token at a time, conditioning on tokens it produced itself. Standard supervised knowledge distillation trains the student on a fixed corpus of prefixes, either ground-truth text or teacher-generated text, so the contexts seen during training differ systematically from the contexts visited at inference. Once the small student makes an early mistake it drifts into territory it never trained on, and errors cascade through the rest of the rollout. This is the same compounding-error phenomenon that makes behavioral cloning quadratic in the horizon under the expert's state distribution. At the same time, the forward-KL objective that supervised KD uses is mass-covering: wherever the teacher puts probability, the student is forced to put some probability too. When the student is too small to represent the teacher faithfully, that pressure smears mass onto low-probability tokens and produces degraded, sometimes hallucinated, continuations. Existing fixes address only one of the two failures. Sequence-level KD still trains on teacher-generated, fixed data and collapses the full distribution to a single beam-search mode. DAgger and ImitKD move toward on-policy data but keep the forward-KL loss and use only the teacher's chosen action, discarding the rich token-level distribution a queryable teacher can provide. Sequence-level reverse-KL methods like MiniLLM choose a better divergence direction but backpropagate through sampling, which is high-variance and requires a large bag of stabilizing tricks. What is needed is a way to train on the student's own generated prefixes using the teacher's full token-level distribution, while being free to choose a divergence direction that respects the student's limited capacity.

The method is Generalized Knowledge Distillation, or GKD. It introduces two orthogonal knobs and combines them into a single objective. The first knob controls the data distribution. With probability λ the batch is built by sampling inputs, generating completions from the student itself, and then asking the teacher to provide its full next-token distribution at every prefix along those self-generated sequences. The gradient is stopped at the sampling step, so the generated sequences are treated as fixed training data for this update. This removes the high-variance policy-gradient term while still giving the student the benefit of training on its own inference-time distribution of contexts. The second knob controls the divergence. Instead of locking the loss to forward KL, GKD uses the generalized Jensen-Shannon divergence with mixture M = β p_T + (1−β) p_S. The small-β limit recovers the mass-covering forward KL KL(p_T || p_S); by symmetry the large-β limit recovers the mode-seeking reverse KL KL(p_S || p_T); and β = 0.5 gives the symmetric Jensen-Shannon divergence. Because JSD is bounded even when the two distributions have disjoint support, early training is better behaved than with raw reverse KL. Putting the two knobs together gives the objective L_GKD(θ) = (1−λ) E_{(x,y)~(X,Y)}[ D(p_T || p_S^θ)(y|x) ] + λ E_{x~X}[ E_{y~p_S(·|x)}[ D(p_T || p_S^θ)(y|x) ] ], where the inner divergence is averaged over the completion tokens and D is the chosen member of the JSD family. Supervised KD is the corner with λ = 0 and forward KL; pure on-policy distillation is the corner with λ = 1 and forward KL. In practice the student should already be supervised-fine-tuned before GKD so that its early generations are meaningful enough for the teacher to give useful feedback.

Because GKD only consumes student samples and a per-sequence divergence, it composes cleanly with reward fine-tuning: one can trade off a task reward against the distillation term, with the distillation regularizer anchoring the student to the teacher rather than to its own initial policy. The implementation below follows the canonical TRL-style trainer path. The loss applies temperature scaling to both logit tensors, computes the generalized JSD with explicit branches for β = 0 and β = 1, and uses log-sum-exp for numerical stability when forming the mixture. Logits are shifted so that the position-n logit predicts token n+1, and only completion positions are kept in the average. The training step optionally replaces the batch with student-generated completions before delegating to the ordinary backward pass.

```python
import random
import torch
import torch.nn.functional as F
from trl.models.utils import unwrap_model_for_generation


def generalized_jsd_loss(student_logits, teacher_logits, labels=None,
                         beta=0.5, temperature=1.0, reduction="batchmean"):
    """Per-token divergence between teacher and student over completion tokens.
    beta: 0 -> forward KL(p_T || p_S), 1 -> reverse KL(p_S || p_T),
    in between -> generalized JSD with mixture M = beta * p_T + (1 - beta) * p_S."""
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs,
                       reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs,
                       reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t),
                         teacher_log_probs + torch.log(beta_t)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs,
                              reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs,
                              reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

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
    student_outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])
    self.teacher_model.eval()
    with torch.no_grad():
        teacher_outputs = self.teacher_model(input_ids=inputs["input_ids"],
                                             attention_mask=inputs["attention_mask"])

    prompt_len = inputs["prompts"].shape[1]
    student_logits = student_outputs.logits[:, prompt_len - 1 : -1, :]
    teacher_logits = teacher_outputs.logits[:, prompt_len - 1 : -1, :]
    labels = inputs["labels"][:, prompt_len:]

    loss = self.generalized_jsd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        beta=self.beta,
    )
    return (loss, student_outputs) if return_outputs else loss


@staticmethod
def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
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
