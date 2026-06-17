I have a model `f(·; w^o)` trained on all of `D`, and someone hands me a forget set `D_f` and
tells me to make the model stop knowing those examples while keeping everything else. The
honest answer — retrain on `D_r = D \ D_f` from scratch — is exactly the thing I am supposed to
avoid, because it costs a full training run per deletion request. So I need to *edit* `w^o`
into some `w^u` that behaves like the retrain would have, cheaply. Let me think about what that
edit can even look like, because the obvious moves all fail in ways that are worth feeling
first.

Start with the cheapest thing imaginable: just keep finetuning on `D_r` and never show the
model `D_f` again. The hope is catastrophic forgetting erodes the forget set's influence on
its own. But there is no force in that loss pushing *against* `D_f` — I am only reinforcing
`D_r`. A network that has memorized class-0 images keeps classifying them correctly long after
I stop training on them, because nothing is actively unlearning them; the retain loss is happy
to leave the forget-set knowledge untouched. So pure finetuning preserves utility and barely
forgets. I need an *active* forgetting pressure.

The textbook active move is gradient ascent on the forget loss — negate the gradient on `D_f`,
climb the cross-entropy there, make the model wrong on those examples. This is NegGrad. And
immediately I can see why it is dangerous: cross-entropy to the true label is unbounded above.
There is no maximum to "be wrong"; the loss on `D_f` can grow forever, so ascent just runs the
weights off to infinity, blowing up the whole model. People patch it into NegGrad+ by mixing in
descent on `D_r`: `L = β·(CE on D_r) − (1−β)·(CE on D_f)`, tuning `β` to balance descending on
retain against climbing on forget. That helps, but the balance is delicate — make `β` too small
and the unbounded ascent term wins, make it too large and the forgetting pressure disappears —
and more deeply, *both* terms are cross-entropy to hard labels. The retain term only defends the
argmax of `D_r`. It says nothing about preserving the *function* the original model computed:
the calibrated confidences, the way it spreads probability across similar classes. I am
protecting the labels, not the behaviour.

Let me look at what the principled crowd does, because they are optimizing a cleaner object.
The indistinguishability line — Golatkar and company — defines unlearning as making the
unlearned weights' distribution close to the retrain-from-scratch distribution, and writes a
Forgetting Lagrangian: the retain loss plus a KL term penalizing the distance from that
retrained reference. They lean on the stability of SGD — if `D` and `D_r` differ by only a few
examples, the two training runs land close in weight space, so unlearning is a small local
correction. Under a local-quadratic approximation that correction has a closed form, and in the
limit it is a Newton step `w − B^{-1}∇L_{D_r}(w)` with a bit of noise, `B` the Hessian. Two
things bother me here. First, `B^{-1}` — forming and inverting a Hessian or Fisher over the
training set scales quadratically in the number of samples, so it dies on anything but tiny
data. Second, and worse, the whole edifice rests on `w^o` being *close* to the retrain
solution. That is fine if `D_f` is a handful of points. But if I am asked to forget an entire
class — ten percent of the data — there is no reason the model trained with that class is near
the model trained without it. The class shaped whole directions of the representation. The
linearization is approximating a small move when the real move is large. The assumption cracks
exactly when the task gets interesting.

So the principled methods buy a guarantee by assuming the edit is small, and pay for it in
scalability and in a premise that fails for big forget sets. I want to drop the premise. Let me
not insist on matching the retrain model at all. Let me instead ask the raw question directly:
what do I actually want `w^u` to *do*?

Two things, and they pull in opposite directions. On `D_r`, I want `w^u` to behave exactly like
`w^o` did — it was correct and well-calibrated there, and I have no reason to change it. On
`D_f`, I want `w^u` to behave *unlike* `w^o` — to lose the specific competence `w^o` had on
those examples. The original model is the thing I am editing, and it is also the thing that
*knows the right answer on `D_r`*. So why am I treating it as a fixed starting point I drift
away from? Let me treat it as a *reference I can selectively agree or disagree with*. Keep a
frozen copy of `w^o`. Call it the teacher. The model I am updating, initialized from `w^o`, is
the student. Now unlearning becomes: train a student that *obeys the teacher on `D_r` and
disobeys it on `D_f`*.

This reframing already pays off at initialization. The student starts *as* the teacher, so it
is immediately good on `D_r` — one of my two goals is satisfied for free before I take a single
step. I only have to work on the disobedience, and protect against collateral damage to the
obedience.

Now I need a way to measure "obey" and "disobey." Hard labels are too blunt — I argued that
above with NegGrad+. The teacher's full output distribution is the thing I want to match (on
`D_r`) or escape (on `D_f`), because that distribution is where the model's real knowledge
lives. There is a well-understood fact about softmax outputs: the top class hides most of the
information, and the *ratios among the small probabilities* — how much mass a "2" puts on "3"
versus "7" — encode a rich similarity structure, the model's learned sense of which classes
look alike. Hinton's distillation exploits exactly this by softening the softmax with a
temperature `T`,

    q_i = exp(z_i / T) / Σ_j exp(z_j / T),

so that raising `T` flattens the distribution and exposes those small-probability ratios
("dark knowledge"). If I want the student to truly replicate the teacher's *function* on `D_r`,
I should match these softened distributions, not just the argmax. And the natural way to
measure agreement between two distributions is the KL divergence. So define, for an example `x`,

    d(x; w^u) = KL( p(f(x; w^o)) ‖ p(f(x; w^u)) )

— the KL from the teacher's softened output distribution to the student's, with the dependence
on `w^u` made explicit because I will optimize the student while the teacher stays frozen. This
is one scalar per example that says how far the student's behaviour has drifted from the
teacher's on that example. Small `d` means obeying; large `d` means disobeying.

Given that distance, the forgetting goal writes itself. To make the student disobey on `D_f`, I
*maximize* `d` there — push the student's forget-set distribution away from the teacher's:

    min_{w^u}  − (1/N_f) Σ_{x_f ∈ D_f} d(x_f; w^u).

And notice what this changes relative to NegGrad. I should be precise: `KL(teacher ‖ student)`
is not globally upper-bounded over all possible student logits; if the student drives probability
to zero where the softened teacher still has mass, the number can still grow. But the forget
signal is no longer a one-hot command to make the true-class probability vanish at any cost. It
is a teacher-referenced divergence between softened output distributions, computed in logit
space against a frozen finite teacher. I am pushing the student away from the teacher's whole
belief vector, not climbing raw cross-entropy against a hard label indefinitely. That is the
controlled pressure I need.

Can I just do that maximization and stop? Let me think about the shared representation. `D_f`
and `D_r` are not processed by disjoint weights — the lower layers compute features used by
every class. When I push the student away from the teacher on forget examples, those gradients
flow back through the shared features and perturb the student's behaviour on retain examples
too. There is no wall in the network separating the two. So I expect maximizing `d` on `D_f`
alone to degrade `D_r` as a side effect — the forget pressure leaks into the retain set through
the shared trunk. I should not have to run it to know this will happen; it is forced by the
fact that the same weights serve both sets. The forget-only objective will raise forget error,
yes, but at the cost of the retain performance I was supposed to protect.

So I add a counter-pressure: while pushing away on `D_f`, simultaneously pull *toward* the
teacher on `D_r`. Minimize `d` on the retain set, maximize it on the forget set:

    min_{w^u}  (1/N_r) Σ_{x_r ∈ D_r} d(x_r; w^u)  −  (1/N_f) Σ_{x_f ∈ D_f} d(x_f; w^u).

There is a nice symmetry here — pull close on one set, push away on the other, the same
teacher anchoring both. It has a contrastive flavour: the teacher is the anchor, retain
examples are positives I attract the student toward, forget examples are negatives I repel it
from. And this is exactly where I can pin down why Bad-T's variant of the teacher-student idea
is weaker. Bad-T also distills, but on the forget set it pulls the student *toward a second,
incompetent teacher* — a randomly initialized network. Pulling toward a random teacher means
pulling toward a near-uniform distribution, which is a weak and noisy target: uniform is just
one particular wrong answer, and the random network's outputs inject noise that bleeds into the
shared features and hurts utility. Pushing *away from the one good teacher* is a sharper signal
— it says "be anything but what the knowledgeable model would say here" — and it uses a single,
clean reference for both directions rather than importing randomness.

One more reinforcement on the retain side. The KL term keeps the student near the teacher's
*distribution*, which is good, but it inherits whatever small errors the teacher had — if the
teacher was slightly wrong on some `D_r` example, matching it faithfully reproduces that error.
I have the true labels for `D_r`; I should use them as an independent incentive to be
*correct*, not merely teacher-like. So add the ordinary task loss (cross-entropy to the real
labels) on the retain set, weighted separately. The objective becomes

    min_{w^u}  (α/N_r) Σ_{x_r} d(x_r; w^u)
             + (γ/N_r) Σ_{(x_r,y_r)} ℓ(f(x_r; w^u), y_r)
             − (1/N_f) Σ_{x_f} d(x_f; w^u),

with `ℓ` cross-entropy and `α, γ` scalars. The KL term is the "stay as the teacher was"
regularizer; the cross-entropy term is the "and actually be right" anchor; together they hold
the retain set, while the negated KL erodes the forget set. I expect the cross-entropy term to
give a small but consistent guard on retain quality — it is doing independent work from the KL,
so removing it should cost a little retain performance.

Now, how do I actually optimize this? The instinct is to just throw all three terms into one
loss and run a stock optimizer. Let me think about what that does dynamically. On any given
minibatch I am asking the student to move *toward* the teacher on the retain points and *away*
from it on the forget points, at the same time, through shared weights. Those two demands
interfere — the same parameter gets a "stay" gradient from one set and a "leave" gradient from
the other, and the resultant thrashes. This is the classic signature of a min-max objective:
two sub-objectives that fight, producing oscillation in the loss rather than smooth descent. I
have seen this shape before — it is the same difficulty GANs have, where naively co-training the
generator and discriminator oscillates, and the standard fix is not to take them in lockstep
but to give one several steps before updating the other.

So borrow that recipe. Don't interleave the max and min at the gradient level; *alternate them
at the epoch level*. Do a full epoch of the maximization on the forget set — the max-step,
pushing the student away from the teacher on `D_f` — then a full epoch of the minimization on
the retain set — the min-step, pulling it back toward the teacher and toward the true labels on
`D_r`. Alternate these. Each phase gets to make coherent progress before the other one pulls in
its direction, which damps the thrash.

But I have to be careful about how many max-steps I do, and this is where the dynamics get
subtle. Each max-epoch raises the forget error (good) but also nudges the retain set off
(through the shared trunk); each min-epoch restores retain (good) but, because it is pulling the
student back toward the teacher, can also *pull the forget set back toward the teacher* — i.e.
partially re-learn what I just forgot. So if I keep alternating forever with equal weight, the
two phases fight to a stalemate and I never reach a point with both high forget error and low
retain error simultaneously. If I do too *few* max-steps, the later min-only phase drags the
forget error back down — I undo my own forgetting. If I do too *many*, I overshoot and wreck
the retain set beyond what the min-steps can repair. There is a window.

The fix that respects this: do the alternation only for the first few epochs — a small number
of max-steps, call it `msteps` — and then stop maximizing and run *only* min-steps for the
remainder. The early max-steps inject the forgetting; the trailing min-only steps restore the
retain performance that the max-steps disturbed, *without* re-teaching the forget set, because
once the max-steps stop there is no longer any pressure pulling the forget set back toward — no,
wait: the min-steps only touch `D_r`, so they pull the *student* back toward the teacher on
retain examples, and only affect the forget set through the shared trunk, which is a much weaker
coupling than the direct max-pressure. So after a couple of max-epochs the forget error is high,
and the trailing min-epochs clean up the retain set while the forget error stays elevated.
Concretely the schedule is: while the epoch counter is below `msteps`, do a max-epoch then a
min-epoch; afterward, do only min-epochs. Stop when the forget error has risen without the
retain error being harmed — which, in practice, happens after only a handful of epochs.

Let me make the KL distance concrete enough to code, because the temperature interacts with the
gradient scale in a way I should not get wrong. Using softened softmaxes at temperature `T`, the
student log-probabilities are `log_softmax(z_s / T)` and the teacher probabilities are
`softmax(z_t / T)`. The KL `d(x) = KL(teacher ‖ student) = Σ_i p_t,i (log p_t,i − log p_s,i)`.
Now the subtlety: in the high-temperature limit, the gradient of this soft-target term with
respect to the student logits scales like `1/T²` — softening the targets shrinks the gradients
quadratically in `T`. Hinton's point is that if you mix this soft term with a hard-label term
(whose gradient does not shrink with `T`), the soft term's contribution vanishes as you raise
`T` unless you compensate. So multiply the distillation term by `T²`, which restores the
gradient magnitude to be comparable across temperatures and keeps the relative weight of the
soft and hard objectives stable when `T` changes. That gives the implementation form

    d(x) = T² · KL( softmax(z_t / T) ‖ softmax(z_s / T) ),

computed with `F.kl_div(log_softmax(z_s/T), softmax(z_t/T))` and a `batchmean` reduction so the
scale is per-example. For the temperature itself, `T = 4` sits in the range where softening
exposes the useful inter-class structure without flattening the distribution into noise — for
small networks, temperatures around 2.5-4 are where distillation works best, higher just blurs
everything toward uniform.

What about the weights `α` and `γ`, and the optimizer? The retain set has two guards on it: the
KL-to-teacher (weight `α`) and the cross-entropy-to-labels (weight `γ`). The cross-entropy is
the one that directly enforces correctness, so it should dominate — I want `γ` large and `α`
small, the KL acting as a light "don't drift" regularizer on top of a strong "be right" signal.
Defaults around `γ ≈ 0.99`, `α ≈ 0.01` reflect that ordering: the retain task loss carries the
weight, the teacher-anchor KL is a gentle correction. The objective is not very sensitive to
these as long as neither is zeroed out — setting `α = 0` loses the soft anchor, setting `γ = 0`
loses the correctness incentive, but across a broad range of nonzero values the retain error
stays low and the forget/test behaviour is what I want. For the optimizer, a small learning
rate (around `5e-4`, with weight decay and momentum, or Adam) is important precisely because of
the min-max instability: decaying the learning rate is what keeps the oscillation between the
push and pull phases under control. The forget and retain sets can even use different batch
sizes to tune how many max- versus min-iterations happen per epoch — another lever on the
balance between the two directions.

There is a final wrinkle for one of the application regimes. By construction this drives the
forget error as *high as possible* — the max-steps push the student maximally away from the
teacher on `D_f`. For removing biases or correcting mislabeled data that is exactly right: I
want the model to *never* reproduce the unwanted behaviour. But for privacy it is wrong in a
specific way. A membership-inference attacker fits per-example confidence distributions from
shadow models trained with and without an example and thresholds the likelihood ratio; an
example the model is *abnormally* bad at is a giveaway that it was deliberately forgotten. So
maximal forget error is itself a privacy leak — it makes the deleted examples *more*
identifiable, not less. For privacy I want the forget error only *as high as a model that never
saw `D_f`* would naturally have, no higher. I cannot get that reference by retraining without
`D_f` (that is the cost I am avoiding). But I can approximate it: build a held-out validation
set drawn from the *same distribution* as the forget set (e.g. if `D_f` is all of class 0, hold
out other class-0 examples), and measure the trained student's error on it. Any correct
predictions there come purely from the model's generalization, not from having memorized those
specific examples — so that validation error is a proxy for "the error of a model that never saw
this distribution." Then, since I checkpoint the student every epoch as the forget error climbs,
I *rewind* to the checkpoint whose forget error is closest to that validation reference: high
enough to have forgotten, not so high as to be conspicuous. That is the rewinding extension —
the same core algorithm, just stopping at a calibrated point rather than at maximal forgetting.

Let me write the core update as it runs inside the harness — one step gets a retain minibatch, a
forget minibatch, a shared optimizer, and the step/epoch counters. The teacher is captured once
(a frozen deep copy of the model as it was before any unlearning). The max-step runs only while
the epoch counter is below `msteps`; the min-step runs every epoch.

```python
import copy
import torch
import torch.nn.functional as F


class UnlearningMethod:
    """SCRUB: a frozen teacher = the original pre-unlearning model; a student,
    initialized from it, is pushed AWAY from the teacher on the forget set
    (max-step) and pulled TOWARD it on the retain set while staying correct
    there (min-step). Max-steps run only for the first `msteps` epochs; min-steps
    run every epoch and restore retain performance the max-steps disturbed."""

    def __init__(self):
        self.msteps = 2        # epochs of max-step before switching to min-only
        self.kd_T = 4.0        # distillation temperature
        self.alpha = 0.01      # weight on KL(teacher || student) on the retain set
        self.gamma = 0.99      # weight on cross-entropy to true labels on the retain set
        self.teacher = None    # frozen copy of the original model, captured lazily

    def _kd_kl(self, student_logits, teacher_logits):
        # d(x) = T^2 * KL( softmax(teacher/T) || softmax(student/T) ).
        # The T^2 restores the soft-target gradient magnitude, which scales as 1/T^2,
        # so the distillation term keeps a stable weight as the temperature changes.
        T = self.kd_T
        p_s = F.log_softmax(student_logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

    def _capture_teacher(self, model):
        # Freeze the original model as the teacher; the student keeps being updated.
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        if self.teacher is None:
            self._capture_teacher(model)            # student starts as the teacher

        retain_x, retain_y = retain_batch
        forget_x, _ = forget_batch                  # max-step ignores forget labels

        # ---- Max-step on the forget set (only for the first `msteps` epochs) ----
        # Push the student AWAY from the teacher on D_f: minimize -KL, i.e. maximize KL.
        # The forget signal is soft teacher disagreement, not hard-label CE ascent.
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

        # ---- Min-step on the retain set (every epoch) ----
        # Pull the student TOWARD the teacher (alpha * KL) and keep it CORRECT
        # on the true labels (gamma * CE); this also restores retain performance
        # disturbed by the max-step through the shared representation.
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

Tracing the causal chain back: retraining is the only sure way to forget but is the cost I must
avoid; finetuning forgets too weakly because nothing pushes against `D_f`; gradient ascent on
the forget loss pushes but is unbounded and explodes; the principled Newton/Fisher scrubs give a
local bound but assume the edit is small and need a Hessian inverse, both of which break for a
class-sized forget set. Reframing the original model as a *frozen teacher* and the model-being-
edited as a *student that selectively obeys it* makes the retain goal free at initialization and
turns "forget" into "disagree with a knowledgeable reference" — measured by KL on softened
outputs, which captures the full learned function rather than just labels.
Maximizing that KL on `D_f` forgets but leaks into `D_r` through shared weights, so I add a
minimization (KL plus task cross-entropy) on `D_r` to hold the line; the resulting min-max
oscillates, so I run it GAN-style — a few alternating max/min epochs to inject forgetting, then
min-only epochs to restore retain — and, when privacy demands a calibrated rather than maximal
forget error, rewind to the checkpoint matching a same-distribution held-out reference.
