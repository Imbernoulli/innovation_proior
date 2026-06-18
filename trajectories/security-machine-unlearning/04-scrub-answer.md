**Problem (from step 3).** Bad Teacher recovered utility on the visible benchmarks but *collapsed* on
hidden mobilenetv2-fmnist (`retain_acc` 0.0, `forget_acc` 1.0, `unlearn_score` 0.1009): the random forget
teacher is a noisy target whose noise corroded the shared trunk faster than the competent teacher could
repair it. The forget signal must stay bounded but become *clean* — no second random network.

**Key idea.** SCRUB: keep one frozen teacher = the original model, and use it for *both* directions. On
`D_r` pull the student *toward* the teacher (`alpha`·KL) and toward the true labels (`gamma`·CE); on `D_f`
push the student *away* from the teacher (maximize KL, i.e. minimize `-KL`). To avoid the min-max thrash,
*alternate*: run the max-step on the forget set only for the first `msteps` epochs (inject forgetting),
then run min-steps only (restore retain without re-teaching the forget set).

**Why it fixes step 3.** "Be anything but what the knowledgeable model would say here" is a sharp,
teacher-referenced, *finite* forget signal that imports no randomness into the shared features — so it
cannot trigger Bad Teacher's noise-driven collapse. The scheduled max-then-min damps the GAN-style
oscillation and lets the trailing min-only epochs repair the retain damage the early max-steps caused.
Expect parity-or-better on the visible benchmarks and, decisively, *no collapse* on mobilenetv2-fmnist.

**Hyperparameters.** `msteps = 2`, `kd_T = 4.0`, `alpha = 0.01`, `gamma = 0.99` (authors' VGG settings).
The KD KL carries the `T^2` gradient-scale factor; teacher captured lazily, frozen, `eval()`. The fixed
`Adam(lr=0.001)` is the small step the min-max instability needs.

```python
import copy

class UnlearningMethod:
    """SCRUB: min-max KL distillation vs a frozen original model.

    Paper: https://arxiv.org/abs/2302.09880
    Reference code: https://github.com/meghdadk/SCRUB
    """

    def __init__(self):
        # Defaults from the authors' VGG notebook.
        self.msteps = 2        # number of max-step epochs (rewind)
        self.kd_T = 4.0        # KD temperature
        self.alpha = 0.01      # weight on KL(student || teacher) in min step
        self.gamma = 0.99      # weight on CE(student, y) in min step
        self.teacher = None    # lazily captured on first step

    def _kd_kl(self, student_logits, teacher_logits):
        # KL(student || teacher) with temperature, as in Hinton KD.
        T = self.kd_T
        p_s = F.log_softmax(student_logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)

    def _capture_teacher(self, model):
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        if self.teacher is None:
            self._capture_teacher(model)

        retain_x, retain_y = retain_batch
        forget_x, _ = forget_batch

        # ---- Max step on forget set (only during the first msteps epochs) ----
        forget_kl_val = 0.0
        if epoch < self.msteps:
            optimizer.zero_grad()
            s_forget = model(forget_x)
            with torch.no_grad():
                t_forget = self.teacher(forget_x)
            forget_kl = self._kd_kl(s_forget, t_forget)
            (-forget_kl).backward()
            optimizer.step()
            forget_kl_val = forget_kl.item()

        # ---- Min step on retain set (every epoch) ----
        optimizer.zero_grad()
        s_retain = model(retain_x)
        with torch.no_grad():
            t_retain = self.teacher(retain_x)
        retain_ce = F.cross_entropy(s_retain, retain_y)
        retain_kl = self._kd_kl(s_retain, t_retain)
        loss = self.gamma * retain_ce + self.alpha * retain_kl
        loss.backward()
        optimizer.step()

        return {
            "loss": float(loss.item()),
            "retain_ce": float(retain_ce.item()),
            "retain_kl": float(retain_kl.item()),
            "forget_kl": float(forget_kl_val),
        }
```
