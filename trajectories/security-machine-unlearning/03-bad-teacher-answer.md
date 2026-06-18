**Problem (from step 2).** NegGrad forgot perfectly (`forget_acc → 0`) but its *unbounded* hard-label
ascent ran the weights off and crashed `retain_acc` (to 0.0101 on vgg16bn), sinking `unlearn_score` below
the passive baseline; its MIA AUC even dipped below 0.5 — the confidently-wrong fingerprint. The
forgetting signal must be *bounded* and anchored to a *reference behavior* (generalization-level
uncertainty), not "move away from the true label" with no destination.

**Key idea.** Dual-teacher distillation. Keep two frozen teachers: a **competent** teacher (a copy of the
original model) and an **incompetent** teacher (a same-architecture network re-initialized to random
weights). Per sample, distill the student toward the competent teacher if it is a retain sample and toward
the incompetent teacher if it is a forget sample —
`KL(student ‖ l_u·incompetent + (1-l_u)·competent)`. The student starts at the competent weights, so
retain utility is held for free; the random forget target is *bounded* and *uninformative*, so the forget
set decays to untrained randomness, not to confident wrongness.

**Why it fixes step 2.** KL toward a fixed finite target has a minimum, so the forgetting cannot diverge —
no weight blow-up, so `retain_acc` recovers toward the ceiling. The random teacher's output is
generalization-level uncertainty by construction, so the MIA AUC should sit near 0.5 instead of NegGrad's
conspicuous dip. Retain and forget are fed *together* (concatenated) so the competent teacher holds the
shared trunk while the random target erases the specific class.

**Hyperparameters.** `KL_temperature = 1.0` (no softening needed; pure distillation, no `T^2` term).
Teachers captured lazily on the first step; incompetent teacher re-randomized with the harness's own
Kaiming init. Default `mean` KL reduction; optimizer/batch/epochs fixed by the harness.

```python
import copy
import torch.nn as nn

class UnlearningMethod:
    """Bad Teacher: dual-teacher KD with competent + incompetent teachers.

    Paper: https://arxiv.org/abs/2205.08096
    Reference code: https://github.com/vikram2000b/bad-teaching-unlearning
    """

    def __init__(self):
        self.KL_temperature = 1.0
        self.competent = None       # = frozen original model
        self.incompetent = None     # = randomly re-initialised same-arch model

    def _freeze(self, m):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    def _random_reinit(self, m):
        # Kaiming init identical to initialize_weights() in run_unlearning.py.
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
        # Ref: UnlearnerLoss in vikram2000b/bad-teaching-unlearning.
        T = self.KL_temperature
        f_t = F.softmax(full_teacher_logits / T, dim=1)
        u_t = F.softmax(unlearn_teacher_logits / T, dim=1)
        lbl = is_forget.view(-1, 1).float()
        target = lbl * u_t + (1.0 - lbl) * f_t
        log_s = F.log_softmax(student_logits / T, dim=1)
        return F.kl_div(log_s, target, reduction='batchmean')

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        if self.competent is None:
            self._capture_teachers(model)

        retain_x, _ = retain_batch
        forget_x, _ = forget_batch

        # Balanced mini-batch: concatenate retain + forget samples.
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
