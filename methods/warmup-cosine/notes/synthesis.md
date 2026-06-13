# Synthesis — warmup + cosine schedule

## What the method IS (identified from the task edit + primary sources)
The "warmup_cosine" baseline is the field-standard fusion of two published ingredients:
1. **Linear (gradual) warmup** — Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training
   ImageNet in 1 Hour" (arXiv:1706.02677). Start at a small lr, ramp *linearly* to the target
   over the first few epochs (they use 5), then resume the normal schedule. They contrast this
   with *constant* warmup (low constant lr then a jump), which spikes; the gradual ramp avoids
   the spike. (Their warmup ramps from eta to k*eta in the large-batch setting; the single-machine
   recipe ramps from a small fraction of base_lr up to base_lr.)
2. **Cosine annealing** — Loshchilov & Hutter 2017, SGDR (arXiv:1608.03983), eq. 5:
   eta_t = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi * T_cur / T_i)).
   The "warmup_cosine" baseline uses a *single* cosine run (T_mult=1, no restarts), eta_min=0,
   eta_max=base_lr, T_i = total_epochs - warmup.

Concrete task baseline (authoritative edit):
    warmup = 5
    if epoch < warmup:    return base_lr * (epoch + 1) / warmup           # linear warmup
    progress = (epoch - warmup) / (total_epochs - warmup)
    return base_lr * 0.5 * (1 + cos(pi * progress))                       # cosine to 0
i.e. epochs 0..4 ramp base_lr/5 -> base_lr (steps of base_lr/5), then cosine base_lr -> 0 over
the remaining (total - warmup) epochs, with the cosine clock starting at end of warmup
(warmup_prefix=True). At epoch 5: progress=0 -> base_lr; at the final epoch progress->1 -> 0.

## The problem (in-frame, pre-method)
Training a deep net with a fixed pipeline (SGD+momentum, fixed base_lr=0.1, weight decay,
200 epochs). The lr is the single most important knob. A constant lr is wrong both ways: at the
*start* a full lr is too large (the net is in a sharp, high-curvature region right after random
init, and a too-large step overshoots / spikes), and at the *end* a full lr is too large to
settle into a minimum (it rattles at the noise floor). Need an lr *schedule* eta(t) over the run.

## Load-bearing ancestors (background) and their gaps
- **Step decay** (He et al. 2016, ResNet; also the WRN recipe of Zagoruyko & Komodakis 2016 used
  by SGDR): hold lr constant, divide by 10 (or 5) at hand-picked milestones (e.g. 30/60/80 % of
  training; WRN: 0.2x at 60/120/160). Works, but (a) the milestones are extra hyperparameters
  tied to the epoch budget; (b) each drop is an *abrupt* discontinuity that shocks the dynamics;
  (c) it does nothing about the *start* — the first step already uses the full lr.
- **Momentum / Nesterov SGD** (Sutskever et al. 2013; Nesterov 1983): the fixed optimizer here.
  Velocity v carries history; if the lr suddenly jumps, the stale v is mismatched (Goyal's
  "momentum correction" eta_{t+1}/eta_t). Relevant: smooth schedules are gentler on momentum.
- **Restarts in optimization** (O'Donoghue & Candes 2012; conjugate-gradient restarts
  Fletcher-Reeves / Powell): the lineage SGDR generalizes — periodically reset to escape
  ill-conditioning. The single-run cosine drops the restart.
- **Cyclical learning rates** (Smith 2015/2017): triangular ramps; related-in-spirit precursor
  to the cosine bump. The one_cycle baseline is the other branch.
- The constant-warmup idea (a low constant lr for the first few epochs) is in He et al. 2016
  already; the gap it leaves (the post-warmup spike) is what gradual warmup fixes.

## Design-decision -> why table
- **Why a schedule at all (not constant lr):** stability threshold eta_c ~ 2/lambda_max(H);
  early curvature is high so eta_c is small (full lr overshoots), late you need small steps to
  settle below the gradient-noise floor. One number cannot serve both ends.
- **Why warmup (start small):** right after random init the loss surface is sharp; a full lr
  exceeds eta_c and catapults the loss (spike/divergence). A small lr early keeps the step under
  eta_c while the net moves to a flatter region; then ramp up.
- **Why *gradual/linear* warmup, not *constant* warmup:** a constant low lr followed by a sudden
  jump to the target re-introduces the catapult at the jump (the region still does not admit the
  full lr) -> training error spikes and never recovers (Goyal's direct observation). A linear
  ramp has no discontinuity, so eta tracks the (relaxing) admissible value. Also gentler on the
  momentum velocity (no jump to mismatch the history).
- **Why warmup length ~5 epochs:** long enough for curvature to relax / moment stats to settle,
  short enough to be a small fraction of the budget; results are robust to the exact number
  (Goyal). With base_lr=0.1 over 200 epochs, 5 epochs is 2.5 % — negligible cost.
- **Why ramp to base_lr from base_lr/warmup (not from 0):** starting at exactly 0 wastes the
  first step (no movement) and is discontinuous in slope at t=0+; starting at base_lr/warmup =
  base_lr/5 gives a clean arithmetic ramp base_lr/5, 2*base_lr/5, ..., base_lr over 5 epochs
  (the step-per-epoch is base_lr/warmup). (timm's warmup_lr_init defaults to a small value /0;
  the task uses base_lr/5 implicitly via (epoch+1)/warmup.)
- **Why cosine decay (not step, not linear) after warmup:** smoothness — no milestone
  hyperparameters, no abrupt shocks. Shape: d/dt cos(pi t/T) = -(pi/T) sin(pi t/T) is ~0 at both
  ends and maximal at the midpoint, so the lr lingers near base_lr early (keep exploring/fast
  progress while far from optimum), decays fastest in the middle, and has a long low-lr tail at
  the end (fine-tune as the gradient shrinks toward the noise floor). Step decay's drops are
  abrupt and milestone-dependent; pure linear decay falls too fast early and too slow late.
- **Why end at eta_min = 0:** as the run ends you want the smallest possible steps to settle into
  the minimum; 0 is the natural floor for a single (non-restarting) run. (SGDR keeps eta_min for
  restarts; with no restart, eta_min=0.)
- **Why measure the cosine clock from the *end* of warmup (warmup_prefix):** the cosine should
  span the *post-warmup* budget so that it starts at exactly base_lr (continuous with the top of
  the ramp) and reaches 0 at the final epoch; progress = (epoch - warmup)/(total - warmup) makes
  cos(0)=1 at epoch=warmup and cos(pi)=-1 at epoch=total.
- **Why this is per-epoch:** the harness calls get_lr once per epoch; cosine is evaluated at the
  epoch index. (SGDR/timm evaluate per-batch for a smoother curve; per-epoch is the discretized
  version and matches the fixed pipeline.)

## Final form (lands on real code)
The get_lr function (task harness) + the equivalent torch CosineAnnealingLR / timm
CosineLRScheduler structure. Grounded in:
- timm CosineLRScheduler._get_lr (linear warmup then 0.5(1+cos) cosine, warmup_prefix path)
- torch CosineAnnealingLR closed form eta_min + (base_lr-eta_min)(1+cos(pi t/T))/2
- SGDR eq. 5 and original repo (1+sin)/2 == half-cosine bump.

## Three-source check
1. Primary: Goyal et al. 2017 (warmup) + Loshchilov & Hutter 2017 SGDR (cosine) — both arXiv
   LaTeX sources read in full. (The "warmup_cosine" method is the union of these two.)
2. Background/ancestors: He 2016 step decay + constant warmup; Zagoruyko & Komodakis 2016 WRN
   step schedule; Nesterov/Sutskever momentum; O'Donoghue & Candes 2012 restarts; Smith CLR —
   read from the two papers' related-work sections + bbl.
3. Third-party explainers: emergentmind warmup (curvature/catapult, variance), deeplearningnotes
   + machinelearningmastery cosine shape & formula. Captured in notes/explainers.md.
</content>
