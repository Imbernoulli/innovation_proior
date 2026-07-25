On-policy distillation tries to train a small student language model to match a large teacher by exposing it to the prefixes it will actually generate at deployment. The supervised-learning floor is to train only on dataset prompts and the teacher's outputs, but that ignores the compounding-error problem: when the student produces its own chain-of-thought, it can drift into states the teacher never visited, and a loss that only knows the teacher's behavior on the original data has no recovery signal. The mismatch between the training distribution and the student's own induced distribution is the same disease that makes plain behavior cloning in robotics fail after the first mistake. The bound is sharp: one early disagreement with probability on the order of the per-state error can throw the policy off the demonstration manifold, after which it pays maximal cost for the remaining steps, giving an extra cost that scales quadratically in the horizon rather than linearly.

Existing soft-distillation approaches try to solve this by carrying the teacher's full probability distribution, but before measuring the value of that soft information we need the hard-target baseline. The crudest defensible choice is to discard everything except the teacher's chosen action and train the student to predict that action with plain cross-entropy. This isolates the central axis of how much of the teacher's soft distribution the loss should use, and it anchors the bottom of the ladder that any soft loss must beat. It also answers the state-distribution question in the simplest possible way: the on-policy prefixes are supplied by the trainer's existing mixing mechanism, and the loss only has to decide what label to place on each visited state.

The method is DAgger, short for Dataset Aggregation. In its original imitation-learning form it iteratively rolls out the current policy, labels every visited state with the expert's action, aggregates those pairs into a growing dataset, and retrains a single policy on the aggregate. Refitting on all data seen so far is Follow-The-Leader, which makes the procedure a reduction of imitation learning to no-regret online learning and drives the loss under the policy's own state distribution to a small value.

I realize the on-policy distillation trainer already has the collection-and-refit half of DAgger built in: its `training_step` mixes, with probability `lmbda`, a batch of student-generated prefixes in place of the dataset batch, and this is exactly rolling out the current policy to gather the states it will actually face at deployment. So I leave `training_step`'s state mixing untouched — the `seq_kd` teacher batch first, then the `lmbda` replacement with student rollouts, in that order — and I only replace the labeling rule inside `compute_loss`. `Top1DaggerGKDTrainer` subclasses the GKD trainer, keeps its completion-token alignment (student and teacher forward passes on the chosen batch, logits shifted by `prompt_lengths - 1 : -1`, labels sliced the same way), and swaps the stock generalized-JSD loss for `top1_dagger_loss`: at each completion position it takes the teacher's logits, computes the argmax to get the deterministic expert action, and trains the student to predict that token with cross-entropy. Positions where the label is `-100` are masked out of both the target and the reduction so prompts and padding never contribute, no temperature is applied because a hard target has no soft distribution to sharpen, and the summed loss is divided by the count of valid completion tokens.

This hard-target supervision is the imitation-learning floor. It collapses the teacher's full vocabulary distribution to a one-hot vector at its mode, throwing away dark knowledge and manufacturing sharp gradients wherever the teacher is genuinely uncertain. Under a large teacher-small student gap it is expected to trail soft-distribution losses, especially on long multi-step reasoning chains where the teacher is multi-modal. Its purpose is to establish the baseline and make the next move clear: stop discarding the teacher's distribution and give the student soft per-token targets.

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
