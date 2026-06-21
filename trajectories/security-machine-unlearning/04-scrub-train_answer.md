Bad Teacher did most of what I hoped, and it failed in the one place I flagged as the risk — which is exactly the opening I need. On the two visible benchmarks it recovered the utility NegGrad+ threw away: resnet20-cifar10 `retain_acc` came back to 0.8902 (above even retain finetuning's 0.8758), vgg16bn-cifar100 to 0.5246, with `forget_acc` 0.0 and MIA AUC 0.4809 / 0.4561 near the never-saw-it operating point — so `unlearn_score` climbed to 0.8031 / 0.6895, back above the passive baseline. But the hidden mobilenetv2-fmnist *collapsed*: `retain_acc` 0.0, `forget_acc` 1.0, MIA AUC 0.6974, `unlearn_score` 0.1009 — the model lost all retained accuracy and ended up predicting class 0 on everything. That is the failure mode I named: the random teacher is a *weak, noisy target*, and on an architecture where that noise couples strongly into the shared features it did not settle — the forget signal won and the retain distillation could not hold. The diagnosis is sharp: the *target* on the forget side is the weak link. I need a forget signal that is just as bounded but *clean* — that imports no randomness into the shared trunk at all.

I propose **SCRUB**: keep one frozen teacher, the original model, and use it for *both* directions in a min-max distillation. NegGrad+'s failure was *no target*; Bad Teacher's fix was a bounded but *noisy* target. SCRUB keeps the boundedness and drops the second teacher entirely, using the one clean reference I already have. On $D_r$ I pull the student *toward* the competent teacher (the part that worked); on $D_f$, instead of pulling toward a random network, I push the student *away from* the competent teacher. "Be anything but what the knowledgeable model would say here" is a sharper, cleaner forget signal than "be like this particular random network" — a single, well-defined reference for both directions, importing no randomness into the shared features.

Define the forget distance the way distillation measures behavior — KL on softened outputs, which captures the model's full learned *function* (the inter-class similarity structure in the small probabilities), not just the argmax. For a sample $x$, $d(x) = \mathrm{KL}(T \,\|\, S)$ on temperature-$T$ softmaxes. Small $d$ means the student obeys the teacher, large $d$ means it disobeys, and the forgetting goal writes itself: *maximize* $d$ on $D_f$. Crucially this push is teacher-referenced and *finite* — a divergence between two softened distributions against a frozen finite teacher — not the unbounded hard-label ascent that blew up NegGrad+, and it carries no random network's noise.

But I cannot maximize $d$ on $D_f$ and stop, for the reason that has bitten every rung: $D_f$ and $D_r$ share the trunk, so pushing away on forget examples leaks through the shared features and degrades retain behavior. So I pair it with a counter-pressure on $D_r$: pull *toward* the teacher (minimize $d$), and — because KL-to-teacher only keeps the student *teacher-like*, inheriting any small teacher errors — add ordinary cross-entropy to the *true* retain labels as an independent "and actually be correct" anchor. The retain objective is $\gamma\cdot\mathrm{CE}(S, y) + \alpha\cdot\mathrm{KL}(T \,\|\, S)$ with the cross-entropy carrying the weight ($\gamma = 0.99$) and the teacher-KL a light don't-drift regularizer ($\alpha = 0.01$); the forget objective is the negated KL, $-\mathrm{KL}(T \,\|\, S)$.

The optimization *dynamics* are where this rung earns its strength over a naive combined loss. If I threw all three terms into one loss and stepped, every minibatch would ask the same shared parameters to move *toward* the teacher on retain points and *away* on forget points at once — the demands interfere and the resultant thrashes. That is the classic min-max signature, the same oscillation GANs have, and the standard fix is not lockstep but *alternation*: a max-step on the forget set and a min-step on the retain set as distinct optimizer steps. The number of max-steps matters subtly — each max-epoch raises forget error but nudges retain off through the trunk, while each min-epoch restores retain but, pulling the student back toward the teacher, can partially *re-teach* the forget set. Alternate forever and the two fight to a stalemate. The resolution is to do the max-step only for the first `msteps` epochs to *inject* the forgetting, then run only min-steps for the rest to *restore* retain performance without re-teaching the forget set, because the min-steps touch only $D_r$ and reach the forget set only through the much weaker trunk coupling. A couple of max-epochs forgets; the trailing min-only epochs clean up retain while forget error stays elevated.

The temperature-$T$ KL carries a $T^2$ factor — $d(x) = T^2\cdot\mathrm{KL}(\mathrm{softmax}(T_{\text{logits}}/T) \,\|\, \mathrm{softmax}(S_{\text{logits}}/T))$, implemented as `F.kl_div(log_softmax(student/T), softmax(teacher/T), reduction='batchmean') * T*T` — because the soft-target gradient scales as $1/T^2$, and the $T^2$ restores its magnitude so the distillation term keeps a stable weight. I use $T = 4$, where softening exposes the inter-class structure without flattening to noise, and the defaults `msteps = 2`, `kd_T = 4`, `alpha = 0.01`, `gamma = 0.99` follow the same VGG settings. The teacher is captured lazily on the first step as a frozen deep copy of `model` in `eval()` — no second network, so none of Bad Teacher's randomness enters. The fixed `Adam(lr=0.001)` is exactly the small step the min-max instability wants; a large step would let the oscillation blow up, so the harness's conservative rate is fortunate here.

So the delta from Bad Teacher is targeted at its collapse: where it pulled the forget set *toward a noisy random teacher* — a weak target that corroded mobilenetv2's trunk faster than repair — I push the forget set *away from the one clean competent teacher*, importing no randomness, and separate the push and pull into alternating min/max phases with the max phase capped at `msteps`. The decisive test is the hidden mobilenetv2-fmnist that Bad Teacher collapsed on: if the clean reference and the schedule are the right fix, SCRUB should *not* collapse — `retain_acc` should recover to the high-0.8 range and `unlearn_score` should land near the visible-benchmark level, making this the strongest rung so far: the first method that forgets sharply, preserves utility, *and* stays stable across all three architectures.

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
