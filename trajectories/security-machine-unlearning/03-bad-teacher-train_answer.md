NegGrad+ confirmed the failure I predicted, in exactly the shape that points the way out. It forgot, hard — `forget_acc` went to 0.0 everywhere — but it wrecked the model: on vgg16bn-cifar100 `retain_acc` collapsed to 0.0101, on resnet20-cifar10 to 0.1726, on the hidden mobilenetv2-fmnist to 0.1111, all far below the prior ceiling of 0.5345 / 0.8758 / 0.9373, and worst on the deepest shared-trunk architecture exactly as the unbounded-ascent argument predicted. The `unlearn_score` of 0.5489 / 0.6822 / 0.6910 came in *below* the passive baseline on every benchmark. And on vgg16bn the MIA AUC dropped to 0.3635, well below 0.5 — not better privacy but the confidently-wrong fingerprint, the model so systematically inverted on class 0 that an attacker could read the scrubbing off the abnormality. The lesson is double: the forgetting signal must be (1) *bounded*, so it cannot run the weights off, and (2) anchored to a *reference behavior* — generalization-level uncertainty — so it stops at "looks like it never studied this" instead of overshooting into "confidently wrong."

The reframe that delivers both is distillation. I want a model that behaves a *prescribed* way on $D_f$ and a *prescribed* way on $D_r$, and "make a model behave like a prescribed target distribution on given inputs" is exactly what distillation does — a student is trained to match a teacher's softened output rather than hard labels, copying *whatever* distribution the teacher emits. The piece NegGrad+ ignored is that the teacher need not be *good*; it is just a source of target behavior, and the student will imitate an idiot as faithfully as a sage. Where hard-label ascent says "move away from X" with no destination — hence unbounded — distillation says "move *toward* Y," and a bounded target $Y$ kills the divergence: KL toward a fixed finite distribution has a genuine minimum, so the optimizer settles instead of running off.

I propose **Bad Teacher**: dual-teacher distillation with one competent and one incompetent teacher. The competent teacher $T_s$ is a frozen deep copy of the model as it enters unlearning — it has seen all of $D$. The incompetent teacher $T_d$ is a frozen, randomly re-initialized copy of the *same architecture* — it has seen nothing. The student is the model I actually update, and crucially it *starts as the competent model*, the very weights the harness hands me, so retain utility is satisfied for free at step zero and I only have to work the forget region. Each sample carries an unlearning label $l_u$: 1 if it is in $D_f$, 0 if in $D_r$. Per sample I pull the student toward $T_s$ when $l_u = 0$ and toward $T_d$ when $l_u = 1$:

$$\mathcal{L} = (1 - l_u)\cdot \mathrm{KL}(T_s \,\|\, S) + l_u \cdot \mathrm{KL}(T_d \,\|\, S).$$

Because $l_u$ is a hard 0/1 selector, exactly one term is alive per sample, so I fold the selection into a single per-sample target distribution, $\text{target} = l_u\, t_d + (1 - l_u)\, t_s$ from the two teacher softmaxes, and the batched objective collapses to one $\mathrm{KL}(\text{target} \,\|\, \text{student})$. That is not an approximation — for $l_u = 0$ it is exactly $t_s$, for $l_u = 1$ exactly $t_d$ — just an efficient vectorization: build the mixed target once, call KL once.

The choice of a *random* forget teacher rather than a malicious one is what buys the privacy-safe regime by construction rather than by tuning. An untrained network's output on a class-0 image is neither a semantic wrong label nor an inverted anti-fact; it is a largely uninformative distribution close to the generalization-level uncertainty I argued is the correct forgetting target. Distilling the student toward *that* on $D_f$ pushes it toward untrained randomness, exactly the property NegGrad+ could not express. The re-randomization must match the harness's own initializer so the incompetent teacher is the same kind of fresh-init network the harness would have built: Kaiming-normal `fan_out` on Conv2d, constant 1/0 on BatchNorm weight/bias, Kaiming-normal `fan_in` on Linear with zero bias.

There is one subtlety that explains *why* retain and forget must be fed *together* rather than forget alone. The random teacher's noise, if it were the only signal, would bleed through the shared trunk and corrode nearby retained behavior — the same coupling that destroyed NegGrad+. Here that corrosion is checked structurally: a class-0 aircraft shares wings and fuselage with the other aircraft in $D_r$, and on all those retain samples the *competent* teacher is simultaneously holding the student to the right "aircraft-ish" distribution. The forget signal erases the *specific* class-0 competence; the live retain distillation preserves the *generic* features. So I build a balanced minibatch by concatenation — `x = cat([retain_x, forget_x])`, `is_forget = cat([zeros(|retain|), ones(|forget|)])` — and run both teachers under `no_grad` while the fixed Adam owns only the student.

The temperature stays at $T = 1$. The usual reason to raise $T$ is to soften a sharp *competent* teacher so its dark knowledge shows, but here the forget target is already an untrained distribution that needs no softening, and the competent teacher's ordinary probabilities are exactly the retain behavior I want to copy, sharpness and all. With no hard-label term mixed in there is no $T^2$ gradient-scaling to manage. The loss is `F.kl_div(log_softmax(student/T), target)` — student log-probabilities as input, mixed teacher probabilities as target, standard direction — under `batchmean` reduction so the KL is the true per-sample average that the fixed `lr=0.001` Adam is calibrated against.

So the delta from NegGrad+ is precise: where it ascended an unbounded hard-label loss on $D_f$ and ran the weights off, I distill the student toward a *bounded* per-sample teacher mixture — the competent original on retain samples, an untrained random network on forget samples, fed together so the retain distillation holds the trunk while the random target erases the specific class. I expect `retain_acc` to recover dramatically toward the ceiling, `forget_acc` to stay low, and the MIA AUC to sit near 0.5 rather than NegGrad+'s conspicuous 0.3635 dip. The one place I am genuinely unsure is stability: distilling toward near-uniform noise is a *weak, noisy target*, and on an architecture where that noise couples strongly into the retain features it could fail to settle — if any benchmark shows collapsed `retain_acc` and spiked `forget_acc`, that is the random teacher winning, and it would mark the *target itself* as the weak link to fix next.

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
