# Third-party explainer captures (warmup + cosine)

## Warmup rationale (emergentmind.com/topics/learning-rate-warmup, search consolidation)
- At initialization, parameters sit in regions of **high loss-landscape curvature**. There is
  a local stability threshold for gradient descent: eta_c(theta) ~= 2 / lambda_max(Hessian).
  If the step size exceeds eta_c, GD on the local quadratic *diverges* — a "loss catapult":
  loss spikes sharply, then the dynamics self-stabilize by moving to a flatter (lower-curvature)
  region. Warmup keeps eta below eta_c early, while the network is still in the sharp region,
  and ramps it up as curvature collapses.
- Linear warmup: scale the global lr alpha by a factor omega_t that ramps from ~0 to 1 over the
  warmup window, e.g. omega_t = min{1, t / tau} over tau warmup iterations (the document's
  Adam-specific form was omega_t = min{1, (1-beta2)/2 * t}, tau = 2/(1-beta2)).
- "By ramping eta smoothly from near zero, warmup tempers the magnitude of weight updates,
  keeping the optimizer in a stable regime."
- For adaptive optimizers (Adam): early steps have high-variance preconditioner estimates from
  too few samples; warmup is a variance-reduction device letting moment estimates settle. (This
  is the adaptive-optimizer story; for plain momentum SGD the curvature/catapult story is the
  operative one.)

## Why constant-then-jump warmup spikes (Goyal primary, confirmed)
- A *constant* low lr for k epochs then a sudden jump to the high target lr re-introduces the
  catapult at the transition: the network has descended into a region whose curvature still does
  not admit the full lr, and the abrupt jump spikes the training error, which never recovers.
  A *gradual* ramp avoids the discontinuity, so the lr is always near the locally admissible
  value as curvature relaxes.

## Cosine annealing rationale (deeplearningnotes, machinelearningmastery)
- Formula: eta_t = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi * T_cur / T_max)).
- Endpoints: T_cur = 0 -> cos(0)=1 -> eta = eta_max; T_cur = T_max -> cos(pi)=-1 -> eta = eta_min.
- Shape: derivative d/dT_cur [cos(pi T_cur/T_max)] = -(pi/T_max) sin(pi T_cur/T_max), which is ~0
  at both ends and maximal in magnitude at the midpoint. So the lr decays *slowly at the start*
  (stays near eta_max to keep exploring / making fast progress), *fastest in the middle*, and
  *slowly at the end* (a long low-lr tail to fine-tune near the optimum). This is the opposite
  of step decay's abrupt drops.
- vs step decay: step decay's sudden 10x drops shock the training dynamics and depend on
  hand-picked milestones; the smooth cosine transition removes both the shock and the milestone
  hyperparameters, leaving just (eta_max, T_max).
- Reference cosine code (machinelearningmastery):
    lr_min, lr_max = 0.001, 0.1
    return lr_min + 0.5*(lr_max - lr_min)*(1 + np.cos(epoch*np.pi/max_epochs))

## Canonical implementations of warmup+cosine
- PyTorch CosineAnnealingLR closed form: eta_min + (base_lr - eta_min)*(1 + cos(pi*last_epoch/T_max))/2
  ("only implements the cosine annealing part of SGDR, not the restarts").
- timm CosineLRScheduler._get_lr(t):
    if t < warmup_t:  lr = warmup_lr_init + t * (base_lr - warmup_lr_init)/warmup_t   # linear warmup
    else (warmup_prefix=True): t = t - warmup_t                                       # cosine measured from end of warmup
           lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * t_curr/t_i))
  warmup_steps = (base_lr - warmup_lr_init)/warmup_t per group. This is exactly "linear warmup
  then cosine decay", with warmup_prefix controlling whether the cosine clock starts at 0 after
  warmup (the task baseline's `(epoch - warmup)/(total - warmup)` is warmup_prefix=True).

## SGDR original code (loshchil/SGDR, SGDR_WRNs.py)
- Uses (1 + sin(curT))/2 with curT advanced by dt = 2pi/(2*Te) per batch and reset at restarts;
  algebraically the same half-cosine bump as eq. 5 (the LaTeX comment line is
  sin(pi/2 + pi*T_cur/T_i) = cos(pi*T_cur/T_i)). Restart: tt=0; Te *= multFactor.
</content>
</invoke>
