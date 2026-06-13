# Context: learning-rate scheduling and the geometry of the minima SGD finds (circa 2019-2020)

## Research question

Training a deep network with SGD-with-momentum, the schedule of the scalar learning rate `eta_t` is the
lever that most decides final test accuracy, with the optimizer, model, augmentation, batch size, and
weight decay held fixed by convention. Two facts sit in tension. First, *generalization* in deep nets is
tied to the **geometry of the minimum** SGD lands in: wide, flat minima generalize better than sharp,
narrow ones. Second, the prevailing schedules are chosen for *optimization speed* — they decay the rate as
fast as the loss will allow — without any explicit mechanism for steering SGD toward the wide minima that
generalize. The question this work asks: if wide minima generalize better, what does that imply the
learning-rate schedule should *do*, and is the standard practice of decaying early leaving generalization
on the table? Concretely, can a schedule derived from the geometry beat a well-tuned cosine or linear
decay at the same budget, and reach the original accuracy in *fewer* epochs?

## Background

A net with `n` parameters is trained by the stochastic step `x_{t+1} = x_t - eta_t * grad f_t(x_t)`, where
`grad f_t` is the minibatch gradient and `eta_t` the learning rate. The state-of-the-art image and
language models are trained with plain SGD-with-momentum and a *hand-scheduled* `eta_t`, not adaptive
optimizers, so the schedule shape is the object of study. Several frames are in the air.

**Flat vs sharp minima and generalization.** A line of work argues that the *width* of the minimum SGD
settles in predicts how well it generalizes: wide, flat minima are robust to the shift between the
empirical (training) loss surface and the true (test) one, so they generalize better; sharp, narrow
minima do not (Hochreiter & Schmidhuber 1997; Keskar et al. 2017). Large-batch training tends to find
sharper minima and generalizes worse, while small-batch (noisier) training finds flatter ones. Width can
be measured — e.g. the largest perturbation to the weights that keeps the loss within a band, or the local
Hessian spectrum — though such measures are expensive and somewhat metric-dependent.

**SGD noise drives the search.** The stochastic-gradient noise has a scale that, to first order, is
`g ≈ eps * N / (B * (1 - m))` with `eps` the rate, `N` the dataset size, `B` the batch size, `m` the
momentum (Smith & Le 2018). A *higher* learning rate means *more* noise, and higher noise pushes SGD out
of narrow basins and toward wider ones (Jastrzębski et al. 2017). So the learning rate is not only a
step-size for speed; it is the *temperature* of the search, and a high rate is a wide-minimum-seeking
device. This reframes "decay the rate" as "cool the search" — and raises the question of *when* to cool.

**Standard schedules decay early.** Step decay holds a high plateau and divides by a factor at hand-picked
milestones; cosine annealing (Loshchilov & Hutter 2017) decays smoothly along a half-cosine from `base_lr`
to ~0 over the whole budget; linear decay drops the rate linearly to 0. All of these begin lowering the
rate essentially from the start (cosine and linear immediately; step decay after the first plateau). Under
the temperature view, they start *cooling* the search early — before SGD has had much time at high
temperature to find a wide basin. Warmup (Goyal et al. 2017) prepends a short linear ramp to protect the
high-curvature initial region, but the body still decays without a deliberate high-temperature *explore*
phase.

**Cyclical and warm-restart schedules.** Cyclical learning rates (Smith 2015) and cosine annealing with
warm restarts (Loshchilov & Hutter 2017) periodically *raise* the rate, confirming that re-heating helps;
the one-cycle policy (Smith & Topin 2019) uses a single rise-then-fall with a deep tail and argues the
large rate is itself a regularizer. These establish that *high rate = good noise = wide minima*, but they
shape the high-rate region as a ramp or a cycle, not as a sustained plateau chosen by budget.

## Baselines

These are the schedules a geometry-derived schedule is measured against; all share the fixed SGD-with-
momentum substrate and differ only in `eta_t`.

**Step / multi-step decay** (He et al. 2016; Zagoruyko & Komodakis 2016). Hold `eta` constant, multiply by
a fixed factor at milestones (e.g. ×0.2 at 60/120/160 of 200). **Gap:** milestones and factor are
hand-tuned and budget-glued; it cools in discrete shocks; and there is no principled high-temperature
explore phase — the first plateau is high but its length is set by a milestone, not by the
wide-minimum search.

**Cosine annealing** (Loshchilov & Hutter 2017). Smooth half-cosine from `base_lr` to 0 over the budget.
**Gap:** begins decaying immediately; the rate is at its peak only at the single instant `epoch=0`, so the
high-temperature search is vanishingly short — it spends most of the run cooling, finding a minimum quickly
but not necessarily a *wide* one.

**Linear decay** (the strongest simple budgeted baseline; Li et al. 2020 "budgeted training"). Decay `eta`
linearly from `base_lr` to 0 over the budget. **Gap:** like cosine, it cools from the start; under the
budgeted-training analysis it is the best of the simple monotone decays, which makes it the bar to beat,
but it still allots no sustained explore phase.

**Warmup + decay** (Goyal et al. 2017). A short linear ramp then a monotone decay. **Gap:** the ramp only
protects the init; the body still cools early, with no deliberate high-rate plateau.

## Evaluation settings

The pre-existing yardsticks for schedule comparison, held fixed across schedules (same model,
augmentation, weight decay, optimizer, batch size; only `eta_t` differs):

- **CIFAR-10 with ResNet-18**, 200-epoch budget, SGD+momentum, standard augmentation (4-pixel-pad random
  crop + horizontal flip). Metric: test accuracy. Used both at full budget and at reduced budgets to test
  the speed claim.
- **ImageNet with ResNet-50**, standard recipe, as the large-scale image check.
- **Machine translation** (IWSLT'14 DE-EN, WMT'14 EN-DE) with Transformers, as an out-of-domain check
  that a schedule generalizes beyond vision (these typically *require* warmup, so the schedule must
  compose with it). Metric: BLEU.
- **Width measurement** as a diagnostic: at the end of training under each schedule, measure the width of
  the minimum reached (largest weight perturbation keeping loss in a band), to test whether the
  better-generalizing schedule indeed lands in a wider minimum.
- Protocol: identical everything except `eta_t`; the schedule called once per step (or per epoch) to set
  the rate; results reported against the original hand-tuned baselines at the same and reduced budgets.

## Code framework

The schedule plugs into the existing SGD-with-momentum harness used for the decay baselines. The optimizer,
model, loss, data pipeline, and weight decay are fixed; the only undecided piece is the function mapping
the current point in training to a scalar rate, which the loop writes into the optimizer before each update.

```python
def get_lr(step, total_steps, base_lr, config):
    """Return the scalar learning rate for this step/epoch.

    Inputs available before any schedule is chosen:
      step        : how far into training we are (updated each step or epoch)
      total_steps : the total training budget
      base_lr     : the initial / reference (peak) learning rate
      config      : architecture / dataset descriptors, if the schedule wants them
    """
    # TODO: the learning-rate schedule to design, derived from the geometry of the
    #       minima SGD finds (wide minima generalize better -> when to explore vs exploit).
    pass


# existing SGD+momentum loop the schedule plugs into
def train(model, loss_fn, data_loader, optimizer, total_steps, base_lr, config):
    for step in range(total_steps):
        lr = get_lr(step, total_steps, base_lr, config)
        for group in optimizer.param_groups:
            group['lr'] = lr
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()      # SGD+momentum update with this lr
```

The loop supplies the current progress; `get_lr` is where the geometry-derived schedule will live.
