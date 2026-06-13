# Synthesis — One-Cycle (1cycle) learning-rate policy

## Method identity (from the task edit.py)
- MLS-Bench baseline `one_cycle` in task `dl-lr-schedule`.
- edit.py: cosine warmup from base_lr/25 to base_lr over first 30%, then cosine anneal
  from base_lr to base_lr/25 over remaining 70%. pct_start=0.3, div_factor=25, final_div=25.
  This is the **two-phase cosine 1cycle** as in PyTorch `OneCycleLR(anneal_strategy='cos')`
  / fastai `fit_one_cycle`. (Note: the task scaffold operates per-epoch on `base_lr` and
  treats `max_lr == base_lr` because the task fixes SGD lr=base_lr=0.1; the canonical
  paper version ramps to a `max_lr` *above* the initial LR found by the LR range test, and
  cycles momentum. The trace is the canonical PAPER version.)
- Real published reference: **Leslie N. Smith & Nicholay Topin, "Super-Convergence: Very
  Fast Training of Neural Networks Using Large Learning Rates", arXiv:1708.07120.** The
  "1cycle" policy is named and defined there (§Super-convergence). The cyclical-momentum
  recipe + div_factor + momentum range 0.95→0.85 are spelled out in the companion technical
  report Smith 2018 "A disciplined approach to neural network hyper-parameters" arXiv:1803.09820.

## Three sources (all retrieved + read this run)
1. PRIMARY: 1708.07120 LaTeX source — read main text + supplemental appendix in full.
   - 1cycle policy definition (line 217): one cycle smaller than total iters, then let LR
     decrease several orders of magnitude below the initial LR for remaining iters.
   - LR range test: ramp LR from ~0 up linearly; the LR at the test-accuracy peak = max_lr;
     min_lr = max_lr / 3 or 4. Unusual: ResNet-56/CIFAR-10 stayed accurate up to LR=3.
   - "Large learning rates regularize training; other regularization must be reduced to
     keep the balance" — central principle.
   - Hessian-Free simplification to estimate optimal LR (the only real math derivation):
     2nd-order Taylor f(θ) ≈ f(θ0)+(θ-θ0)ᵀ∇f+½(θ-θ0)ᵀH(θ-θ0); finite-difference Hessian
     H(θ)=lim_{δ→0}[∇f(θ+δ)-∇f(θ)]/δ; AdaSecant optimal LR ε*≈(θ_{i+1}-θ_i)/(∇f(θ_{i+1})-∇f(θ_i));
     rewrite via θ_{i+1}=θ_i-ε∇f(θ_i) over THREE sequential iterates:
     ε* = ε (θ_{i+1}-θ_i) / (2θ_{i+1}-θ_i-θ_{i+2}); global LR by summing |numerator| and
     |denominator| (Schaul et al. "No more pesky learning rates" used squares; here abs to
     keep positivity). Small Hessian estimate ⇒ large optimal LR ⇒ wide/flat minima.
   - Intuitive (appendix): loss-topology traversal (Goodfellow et al. 2014) — small LR to
     start (steep early progress), large LR to cross the flat valley fast, small LR at end
     to settle into the trough. CLR = curriculum learning (Bengio 2009) + simulated
     annealing (Aarts 1988).
   - Generalization (appendix): noise scale g ≈ εN/(B(1-m)) (Smith & Le 2017); LR/batch
     ratio controls minima width (Jastrzębski 2017); flat minima generalize (Keskar 2016,
     Hochreiter 1997). SGDR (Loshchilov & Hutter) sawtooth-with-restart does NOT show
     super-convergence per their tests.
2. ANCESTORS:
   - Cyclical LR (Smith 2015 arXiv:1506.01186) — read. triangular policy:
     cycle=floor(1+epochCounter/(2*stepsize)); x=|epochCounter/stepsize - 2*cycle + 1|;
     lr = base_lr + (max_lr-base_lr)*max(0, 1-x). LR range test introduced here.
     triangular2 (halve amplitude each cycle), exp_range (gamma^iter). Motivation: a
     short-term-bad LR increase can be long-term-good (cross saddle points).
   - Disciplined hyperparams (Smith 2018 arXiv:1803.09820) — read §4. 1cycle restated;
     cyclical momentum derived from SGD-with-momentum update v=αv-ε∇L, θ+=v ⇒ momentum
     scales updates like LR; high constant momentum acts like pseudo-increasing LR; so when
     LR is being increased, momentum should be *decreased* (0.95→0.85) to keep stability
     and let new gradients steer; reverse on the way down. WD stays *constant* (not cycled);
     grid-search WD, small amount of overfitting indicates best WD. Momentum range test is
     NOT useful (loss just decreases monotonically with momentum) — that's why momentum is
     set by reasoning about the LR coupling, not by a range test.
3. THIRD-PARTY EXPLAINER: Sylvain Gugger (fastai) "The 1cycle policy" — read via WebFetch.
   Confirms: warmup phase = stability; high-LR middle = regularization (flatter minima,
   smaller Hessian); cyclical momentum 0.95→0.85 (low momentum at high LR so SGD goes in
   new directions, more weight on new gradients); final annihilation phase descends orders
   of magnitude below initial LR to drop into a steeper local min inside the smooth region;
   LR-finder must use same batch size / WD as training.

## Baselines (prior art / what it reacts to)
- Piecewise-constant / step decay: global LR ≈0.1 for many epochs, ÷10 at plateaus, repeat
  2-3×. GAP: each LR phase has the characteristic rise-then-plateau, wastes the bulk of
  iterations on a flat valley making little progress; hand-tuned milestones; uses
  conventionally *small* LR throughout.
- Cyclical LR triangular (Smith 2015): oscillate LR between bounds for many cycles. GAP:
  many full cycles; ends mid-cycle or at min but never drives LR far below the min; doesn't
  by itself produce the order-of-magnitude speedup.
- SGDR cosine warm restarts (Loshchilov & Hutter 2016): cosine decay then jump back to top
  (sawtooth). GAP: the restart jumps don't allow the super-convergence regime per the
  paper's tests.
- Warmup+(step/cosine) (He 2016 / Goyal 2017): linear warmup then decay — "a discretized
  CLR" but only the up-ramp; large-batch stabilization, not a single up-then-down-then-
  annihilate cycle, and momentum left constant.
- Adaptive methods (Adam/AdaGrad/AdaDelta/Nesterov): don't reach the very large LRs that
  give super-convergence; Adam in particular did not show the phenomenon even with 1cycle.

## Design decisions → why
- ONE cycle (not many): the loss-topology picture wants exactly small→large→small once,
  matching converge-begin / cross-valley-fast / settle-into-trough.
- Ramp UP first (warmup): start too large diverges; small LR lets convergence begin and
  makes the steep early progress.
- max_lr from LR range test peak (not guessed): single short run gives the largest LR the
  net tolerates; using it = the regularizing large-LR regime.
- initial_lr = max_lr/div_factor (div_factor≈25 in fastai/PyTorch; paper says min = max/3-4
  for the *cyclic min*; the 1cycle warmup start is lower still). Start the warmup well below
  max so the ramp has range.
- final annihilation to min_lr = initial_lr/final_div (≈orders of magnitude below initial):
  needed to settle into a narrow minimum inside the flat region — "decrease several orders
  of magnitude less than the initial learning rate" (paper).
- cosine vs linear ramps: paper/CLR use linear; fastai/PyTorch default cosine (smoother,
  spends more time near the extremes). edit.py uses cosine — that's the modern default.
  pct_start=0.3: spend 30% going up, 70% coming down + annihilating.
- cyclical momentum 0.95→0.85 inverse to LR: from update algebra momentum acts like LR;
  high momentum + rising LR ⇒ instability and effectively even larger LR; lower momentum at
  high LR keeps it stable and lets fresh gradients steer toward flat regions; restore high
  momentum as LR drops to accelerate the final descent.
- weight decay constant (NOT cycled): WD is the one regularizer that should be a fixed
  grid-searched value; reduced (vs typical) so the large-LR regularization isn't doubled up.
- large LR regularizes: LR-range test shows training loss UP while test loss DOWN over a
  large-LR band ⇒ generalization gap shrinks ⇒ regularizing (per Goodfellow's definition).
  Also g ≈ εN/(B(1-m)): larger ε ⇒ larger SGD noise ⇒ wider/flatter minima ⇒ better
  generalization. So all regularizers (WD, dropout, batch) must be *reduced* to keep the
  balance when LR is large.

## Final code (field-appropriate landing)
- Canonical: PyTorch `torch.optim.lr_scheduler.OneCycleLR` (read). Two phases by default
  (fastai). _annealing_cos(start,end,pct)=end+(start-end)/2*(cos(pi*pct)+1).
  Phase boundaries: initial_lr→max_lr over pct_start; max_lr→min_lr over remainder.
  Momentum cycled inverse via same anneal func, max_momentum→base_momentum then back.
  Defaults: pct_start=0.3, anneal='cos', base_momentum=0.85, max_momentum=0.95,
  div_factor=25, final_div_factor=1e4, three_phase=False. Steps per BATCH.
- Scaffold (pre-method): a per-step LR/optimizer harness with `get_lr(step)` stub returning
  the LR (and a momentum knob), plus an LR-range-test stub. The contribution = the shape of
  get_lr over the run + the momentum coupling.
```
