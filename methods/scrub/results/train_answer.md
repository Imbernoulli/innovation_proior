A model $f(\cdot; w^o)$ has been trained by empirical risk minimization on a dataset $D$, and afterward we are handed a forget set $D_f \subset D$ and told to remove its influence, keeping the retain set $D_r = D \setminus D_f$ intact. We need new weights $w^u$ such that $f(\cdot; w^u)$ no longer behaves as though it was trained on $D_f$, while preserving accuracy on $D_r$ and on held-out data — and we need it cheaply, because the only procedure that provably forgets, retraining from scratch on $D_r$, is exactly the cost we are trying to avoid. The difficulty is that deep networks entangle the two sets in one shared representation: the lower layers compute features used by every class, so $D_f$ and $D_r$ are encoded by the same weights and there is no clean way to excise one without tearing into the other.

The obvious moves each fail in an instructive way. Pure finetuning on $D_r$ — never showing the model $D_f$ again and hoping catastrophic forgetting erodes it — preserves utility but barely forgets, because nothing in the retain loss pushes *against* $D_f$; a memorized class stays classified correctly long after we stop training on it. The textbook active move, gradient ascent on the forget loss (NegGrad, and its retain-mixing cousin NegGrad+ with $L = \beta\cdot\mathrm{CE}_{D_r} - (1-\beta)\cdot\mathrm{CE}_{D_f}$), supplies pressure but is unbounded: cross-entropy to the true label has no maximum, so the ascent term runs the weights off to infinity unless $\beta$ is tuned into a delicate balance, and even when balanced, both terms are cross-entropy to *hard labels* — the retain term defends only the argmax, not the calibrated function the original model computed. The principled indistinguishability line (Golatkar and others) writes a Forgetting Lagrangian and, under a local-quadratic approximation, lands on a Newton/Fisher scrub $w - B^{-1}\nabla L_{D_r}(w)$ plus noise; but forming and inverting the Hessian $B$ scales quadratically in the number of samples, and the whole construction rests on $w^o$ being *close* to the retrain solution — true for a handful of deleted points, false for a whole class that shaped entire directions of the representation. Bad-T's incompetent-teacher distillation pulls the student toward a *random* teacher on $D_f$, but a random network emits near-uniform targets: a weak, noisy signal that injects randomness into the shared features and degrades utility. So we want pressure that is bounded, scalable, makes no closeness assumption, and protects the function on $D_r$ rather than just its labels.

I propose SCRUB (SCalable Remembering and Unlearning unBound), which casts forgetting as a teacher-student problem. The key reframing is to stop treating the original model as a fixed starting point we drift away from and instead treat it as a *reference we selectively agree or disagree with*: keep a frozen copy of $w^o$ as the teacher, and update a student, initialized from $w^o$, that obeys the teacher on $D_r$ and disobeys it on $D_f$. This pays off immediately at initialization — the student starts *as* the teacher, so it is already good on $D_r$, and half the goal is satisfied for free before a single step. What remains is to inject disobedience on $D_f$ and protect the obedience on $D_r$. Agreement is measured not by hard labels — too blunt, as the NegGrad+ argument showed — but by the temperature-softened KL divergence between output distributions, because the softened softmax exposes the ratios among the small probabilities ("dark knowledge"), the model's learned sense of which classes look alike, which is where its real function lives. With $p_T(z) = \mathrm{softmax}(z/T)$, define the per-example distance

$$d(x; w^u) = \mathrm{KL}\big(\,p_T(f(x; w^o))\;\|\;p_T(f(x; w^u))\,\big),$$

one scalar that says how far the student's behaviour has drifted from the frozen teacher's on $x$. Forgetting then writes itself: to make the student disobey on $D_f$ we *maximize* $d$ there; to make it obey on $D_r$ we *minimize* $d$ there. Unlike NegGrad, this is no longer a one-hot command to drive the true-class probability to zero at any cost — it is a divergence between softened distributions against a frozen finite teacher, pushing the student away from the teacher's whole belief vector rather than climbing raw cross-entropy indefinitely. Maximizing $d$ on $D_f$ alone, though, leaks into $D_r$: the forget gradients flow back through the shared trunk and perturb retain behaviour, since the same weights serve both sets. So we add a counter-pressure — minimize $d$ on $D_r$ while maximizing it on $D_f$ — giving the contrastive shape where the teacher is the anchor, retain examples are positives we attract toward, forget examples are negatives we repel from. This is exactly where SCRUB beats Bad-T: pushing *away from the one good teacher* is a sharper, cleaner signal than pulling *toward a random one*, and it uses a single reference for both directions instead of importing noise. One more guard on the retain side: the KL term keeps the student near the teacher's *distribution* but inherits whatever small errors the teacher had, so we add the ordinary task cross-entropy to the true labels of $D_r$ as an independent correctness incentive. The full objective is

$$\min_{w^u}\;\; \frac{\alpha}{N_r}\sum_{x_r \in D_r} d(x_r; w^u)\;+\;\frac{\gamma}{N_r}\sum_{(x_r,y_r)} \ell\big(f(x_r; w^u), y_r\big)\;-\;\frac{1}{N_f}\sum_{x_f \in D_f} d(x_f; w^u),$$

with $\ell$ cross-entropy and $\gamma \gg \alpha$ (defaults $\gamma \approx 0.99$, $\alpha \approx 0.01$): the task loss carries the retain side, the teacher-anchor KL is a light "don't drift" regularizer on top of a strong "be right" signal.

Two further pieces make it actually work. First, the KL distance must be implemented with the right temperature scaling. Using $p_s = \log\mathrm{softmax}(z_s/T)$ and $p_t = \mathrm{softmax}(z_t/T)$, the gradient of the soft-target term with respect to the student logits scales like $1/T^2$, so softening shrinks the gradients quadratically; to keep the distillation term's weight stable across temperatures we multiply by $T^2$, giving

$$d(x) = T^2 \cdot \mathrm{KL}\big(\,\mathrm{softmax}(z_t/T)\;\|\;\mathrm{softmax}(z_s/T)\,\big),$$

computed as $T^2\cdot\mathrm{kl\_div}$ with a $\mathtt{batchmean}$ reduction, at $T = 4$ — the range where softening exposes inter-class structure without flattening everything toward uniform. Second, the optimization itself. Minimizing on $D_r$ and maximizing on $D_f$ through shared weights interferes: the same parameter receives a "stay" gradient from one set and a "leave" gradient from the other, the resultant thrashes, and the loss oscillates rather than descends — the classic signature of a min-max objective, the same trouble GAN training has. So we borrow the GAN fix: instead of interleaving max and min at the gradient level, alternate them at the epoch level, a full max-epoch on $D_f$ then a full min-epoch on $D_r$, each phase making coherent progress before the other pulls back. But the number of max-steps matters, because each min-epoch, by pulling the student back toward the teacher, can partially re-teach what was just forgotten; alternate forever and the two phases fight to a stalemate that never reaches high forget error and low retain error simultaneously. The resolution is to alternate only for the first few epochs — a small number $\mathtt{msteps}$ of max-steps that inject the forgetting — and then run *only* min-epochs for the remainder, which restore the retain performance the max-steps disturbed without re-teaching $D_f$, because once the max-pressure stops the min-steps touch $D_f$ only through the weak shared-trunk coupling. A small, decaying learning rate (around $5\times10^{-4}$) further tames the oscillation, and the forget and retain sets may use different batch sizes to tune the max-to-min iteration ratio. Finally, for the privacy regime there is the SCRUB+R extension: the base algorithm drives forget error *maximal*, ideal for removing biases or correcting mislabels but a liability for privacy, since an abnormally high forget error is itself a tell for a membership-inference attacker. We instead build a held-out validation set drawn from the *same distribution* as $D_f$ — its error under the trained student approximates the error of a model that never saw that distribution — checkpoint every epoch, and rewind to the checkpoint whose forget error is closest to that reference: just high enough to have forgotten, not so high as to be conspicuous.

```python
import copy
import torch
import torch.nn.functional as F


class UnlearningMethod:
    """SCRUB: min-max KL distillation against a frozen original-model teacher."""

    def __init__(self):
        self.msteps = 2        # epochs of max-step before switching to min-only
        self.kd_T = 4.0        # distillation temperature
        self.alpha = 0.01      # weight on KL(teacher || student) on the retain set
        self.gamma = 0.99      # weight on cross-entropy to true labels on the retain set
        self.teacher = None    # frozen copy of the original model, captured lazily

    def _kd_kl(self, student_logits, teacher_logits):
        # d(x) = T^2 * KL( softmax(teacher/T) || softmax(student/T) ).
        # The T^2 compensates the 1/T^2 scaling of soft-target gradients,
        # keeping the distillation term's weight stable across temperatures.
        T = self.kd_T
        p_s = F.log_softmax(student_logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

    def _capture_teacher(self, model):
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        if self.teacher is None:
            self._capture_teacher(model)            # student starts as the teacher

        retain_x, retain_y = retain_batch
        forget_x, _ = forget_batch

        # ---- Max-step on D_f (only for the first msteps epochs): push AWAY ----
        forget_kl_val = 0.0
        if epoch < self.msteps:
            optimizer.zero_grad()
            s_forget = model(forget_x)
            with torch.no_grad():
                t_forget = self.teacher(forget_x)
            forget_kl = self._kd_kl(s_forget, t_forget)
            (-forget_kl).backward()                 # minimize -KL == maximize KL
            optimizer.step()
            forget_kl_val = forget_kl.item()

        # ---- Min-step on D_r (every epoch): pull TOWARD + stay CORRECT ----
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
