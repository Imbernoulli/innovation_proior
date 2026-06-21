We need to remove the influence of a forget set from an already-trained deep classifier without retraining from scratch, which is the expensive but otherwise gold-standard solution. The real task is approximate unlearning: nudge the existing weights so the model behaves like one that was trained only on the retained data. That means full accuracy on the retained set, but on the forget set the model should show generalization-level uncertainty, not confident wrong answers. Confidently misclassifying the forget set is itself a privacy leak, because a membership-inference attacker can notice that those samples were specially treated.

Existing ideas all fall short in ways that matter here. Retraining is correct but too costly per request. SISA requires sharding the data and checkpointing many models before training even begins, so it cannot help an already-deployed monolithic network. Fisher scrubbing and its NTK variants need second-order information or an auxiliary linearized model, and they assume the original model was trained with ordinary SGD. UNSIR works only at the class level and pushes forget accuracy exactly to zero, which is the confidently-wrong regime we want to avoid. NegGrad+ actively ascends the forget cross-entropy, but that signal is unbounded: the ascent keeps producing gradient after the retain loss has saturated, so the weights run off and retain accuracy collapses. What is missing is a bounded forget signal that has a fixed destination, namely the behavior of a model that never learned the forget data at all.

The method I propose is Bad Teacher, also called incompetent-teacher unlearning. It is a selective knowledge-distillation procedure that uses two frozen teachers and routes each sample to the right one. The competent teacher is a frozen copy of the original fully trained model; it supplies the target behavior on retained data. The incompetent teacher is a freshly random-initialized copy of the same architecture, never trained; it supplies the target behavior on forget data. The student is the model we are actually producing, and it is initialized to the original trained weights so that retain utility starts intact and only needs to be preserved, not relearned.

The key insight is that distillation makes the student copy whatever distribution a teacher emits, regardless of whether that teacher is good. On a retain sample we therefore minimize KL divergence from the competent teacher's soft distribution to the student's distribution. On a forget sample we minimize KL divergence from the incompetent teacher's random distribution to the student's distribution. Because the forget target is an untrained output, the student is pulled toward genuine uncertainty rather than toward a confident wrong class. Because retain and forget samples are mixed in the same loader, the competent teacher keeps the shared features useful for retained classes while the random target erases only the specific forget-class information. The per-sample loss is simply a single KL against a target constructed from the two teachers using the unlearning label.

Concretely, for unlearning label l_u equal to 1 on forget samples and 0 on retain samples, the mixed teacher target is l_u times the incompetent teacher's softmax plus (1 - l_u) times the competent teacher's softmax. The student enters the KL as log-softmax. I use temperature T = 1 by default, because the competent teacher's natural probabilities are exactly the retain behavior we want to copy and the incompetent teacher is already diffuse; there is no hard-label term mixed in, so no T-squared scaling factor is needed. The teachers are kept in eval mode and forwarded under no_grad, and only the student receives optimizer updates.

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnlearningMethod:
    """Bad Teacher: dual-teacher distillation with competent + incompetent teachers."""

    def __init__(self, KL_temperature=1.0):
        self.KL_temperature = KL_temperature
        self.competent = None      # frozen copy of the original trained model
        self.incompetent = None    # frozen, randomly re-initialized same-architecture model

    def _freeze(self, m):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    def _random_reinit(self, m):
        # Kaiming-style reinitialization matching standard convnet training defaults.
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)
            elif isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, mode='fan_in', nonlinearity='relu')
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)

    def _capture_teachers(self, model):
        self.competent = copy.deepcopy(model)
        self._freeze(self.competent)

        self.incompetent = copy.deepcopy(model)
        self._random_reinit(self.incompetent)
        self._freeze(self.incompetent)

    def _unlearner_loss(self, student_logits, full_teacher_logits,
                        unlearn_teacher_logits, is_forget):
        T = self.KL_temperature
        f_t = F.softmax(full_teacher_logits / T, dim=1)
        u_t = F.softmax(unlearn_teacher_logits / T, dim=1)
        lbl = is_forget.view(-1, 1).float()
        target = lbl * u_t + (1.0 - lbl) * f_t
        log_s = F.log_softmax(student_logits / T, dim=1)
        # batchmean gives the exact per-sample KL average.
        return F.kl_div(log_s, target, reduction='batchmean')

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        if self.competent is None:
            self._capture_teachers(model)

        retain_x, _ = retain_batch
        forget_x, _ = forget_batch

        # Concatenate retain and forget into one mixed mini-batch.
        x = torch.cat([retain_x, forget_x], dim=0)
        is_forget = torch.cat([
            torch.zeros(retain_x.size(0), device=retain_x.device),
            torch.ones(forget_x.size(0), device=forget_x.device),
        ], dim=0)

        student_logits = model(x)
        with torch.no_grad():
            full_t = self.competent(x)
            unl_t = self.incompetent(x)

        loss = self._unlearner_loss(student_logits, full_t, unl_t, is_forget)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": float(loss.item())}
```
