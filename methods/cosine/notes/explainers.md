# Third-party explainers captured (this run)

## Cosine annealing formula (cross-checked against primary eq.4 and PyTorch closed form)
eta_t = eta_min^i + 0.5*(eta_max^i - eta_min^i)*(1 + cos(pi * T_cur / T_i))
- T_cur = epochs since last restart (can take fractional/per-batch values 0.1, 0.2,...)
- T_i = length of i-th run/cycle
- T_cur=0 -> cos(0)=1 -> eta = eta_max ; T_cur=T_i -> cos(pi)=-1 -> eta = eta_min
PyTorch CosineAnnealingLR closed form (eta_min=0): eta_min + (base_lr-eta_min)*(1+cos(pi*last_epoch/T_max))/2
MLS-Bench baseline = single run, eta_min=0, eta_max=base_lr, T_i=total_epochs, T_cur=epoch:
   base_lr*0.5*(1+cos(pi*epoch/total_epochs))    -- EXACT match.

## Intuition (jeremyjordan.me, zeromathai, timm)
- cosine: smooth decay max->min; spends long time high early (broad/exploratory steps), then accelerating
  drop near the end (fine convergence). Smooth = stable dynamics vs abrupt step drops.
- restarts: drastically increase LR to exit a basin and continue exploring; helps cross saddle-point
  plateaus (small gradients); diversify trajectories -> snapshot ensembles "for free".
- annealing to ~0 at end of each cycle gives fine-grained convergence; the end-of-cycle iterate is the
  recommendation (incumbent).
- exploration->convergence cycle, repeated, vs single explore->converge trajectory.

## Connection back to O'Donoghue & Candes (primary related work, arXiv:1204.3982)
- Accelerated/heavy-ball methods in HIGH-momentum (under-damped) regime ripple; period ~ sqrt(condition number).
- Restart = reset momentum to zero, take current iterate as new start; recovers optimal linear rate
  without knowing mu. Fixed restart interval k* = e*sqrt(8L/mu). Adaptive function/gradient schemes.
- SGDR adapts the *spirit* (periodic restart of a momentum method) to stochastic DL, but emulates the
  restart by raising the LR (warm restart) rather than literally zeroing momentum/iterate.

## Smith CLR (arXiv:1506.01186), closely related cyclical idea
- "increasing LR may have short-term negative effect but long-term beneficial effect"
- triangular policy: linear up then linear down between [base_lr, max_lr], cycle length=2*stepsize.
- saddle-point rationale (Dauphin): raising LR speeds traversal of saddle plateaus.
- CLR oscillates between bounds; SGDR instead anneals max->min along a cosine and restarts to max.

## Step-decay baselines (the prior schedule SGDR reacts to)
- ResNet (He 2015): lr 0.1, /10 when error plateaus (32k,48k iters), terminate 64k. Staircase.
- WRN (Zagoruyko 2016): lr 0.1, *0.2 at epochs 60,120,160; total 200 epochs; SGD+Nesterov, wd 5e-4, mom 0.9.
- pain: piecewise-constant, discontinuous drops, must hand-pick milestones AND total budget;
  no smooth transition; drop timing is a tuned hyperparameter.
