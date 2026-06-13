# Synthesis — cosine annealing (within SGDR), for paper-to-reasoning

## What the MLS-Bench "cosine" baseline IS
The published method is the **cosine annealing learning-rate schedule** introduced in SGDR
(Loshchilov & Hutter, ICLR 2017, arXiv:1608.03983). The general SGDR schedule (eq. 4):
  eta_t = eta_min^i + 0.5*(eta_max^i - eta_min^i)*(1 + cos(pi * T_cur / T_i))
The MLS-Bench baseline is the single-run specialization (no restarts): eta_min=0, eta_max=base_lr,
T_i = total_epochs, T_cur = epoch:
  get_lr(epoch) = base_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))
This is exactly torch.optim.CosineAnnealingLR's closed form with eta_min=0, T_max=total_epochs.
The canonical paper trace must derive the full schedule (cosine within a run + the warm-restart idea),
and land on the code (the single-cycle get_lr + the restart generalization). The reasoning derives the
canonical method; the MLS-Bench scaffold's single-cycle get_lr is the natural "no-restart" instance.

## Pain point (research question)
Training deep nets (WRN/ResNet on CIFAR) is dominated by hand-tuning the LR schedule + weight decay.
The reigning recipe is piecewise-constant step decay (multiply by 0.2 at epochs 60/120/160; or /10 at
plateaus). Problems with step decay:
 - Discontinuous: abrupt drops; the schedule is a staircase with hand-picked milestones.
 - Two coupled tunables that are awkward: WHERE to drop (milestones) and HOW MUCH (factor), AND the
   total budget T must be fixed in advance (the milestones are fractions of T).
 - No good "anytime" behavior: if you stop early you're mid-plateau at a too-high LR; the good solution
   only exists at the very end.
 - Second-order methods (Newton, L-BFGS) ruled out: stochastic grad, ill-conditioning, saddle points,
   intractable inverse Hessian for large n. Adam/AdaDelta exist but the SOTA ResNets used plain
   SGD+momentum with a hand-scheduled LR. So the lever is the *schedule*, not the optimizer.
Goal: a schedule that (a) is smooth, (b) needs few hyperparameters (ideally just eta_max and T),
(c) has good anytime performance, (d) can be made to deliver multiple good solutions during one run.

## Background concepts (load-bearing, pre-method, all knowable before)
1. SGD with (Nesterov) momentum: v_{t+1}=mu*v_t - eta_t grad; x_{t+1}=x_t+v_{t+1}. eta_t decreasing.
2. Step-decay schedule = staircase, the prevailing wisdom (He ResNet; Zagoruyko WRN). The blue/red lines.
3. Restarts in gradient-FREE optimization (CMA-ES line; Loshchilov-Schoenauer-Sebag 2012; Hansen 2009):
   for multimodal functions, restart the search, often increasing population lambda (e.g. doubling) at
   each restart for good anytime/global behavior. Start small lambda, double after each restart.
4. Restarts in gradient-BASED optimization:
   - conjugate gradient flushed every n iters (Fletcher-Reeves 1964); Powell 1977 restart test on
     orthogonality of consecutive gradients.
   - O'Donoghue & Candes 2012 (arXiv:1204.3982) — THE key intellectual ancestor. Accelerated/heavy-ball
     gradient methods in high-momentum (under-damped) regime exhibit periodic RIPPLES in f; the ripple
     period is proportional to sqrt(local condition number) (psi_mu ~ sqrt(mu/L)). Restarting (reset
     momentum to 0, current iterate as new start) recovers the optimal linear convergence rate WITHOUT
     knowing mu. Fixed restart interval optimum k* = e*sqrt(8L/mu) gives O(sqrt(L/mu) log(1/eps)).
     Adaptive function-scheme (restart when f increases) and gradient-scheme (restart when
     grad·step>0). KEY: restart = momentum reset = a fresh acceleration phase; the condition number is
     unknown and varies LOCALLY, so periodic restart is the robust answer.
5. Smith CLR 2015 (arXiv:1506.01186) — cyclical LR; "increasing LR may hurt short-term but help
   long-term"; triangular up/down between [base_lr,max_lr]; saddle-point traversal rationale
   (Dauphin et al.: difficulty is saddle plateaus, not local minima; raising LR speeds escape). CLR is
   "closely related in spirit" but does NOT focus on restarts and oscillates linearly between bounds.
6. Snapshot diversity: a periodic-restart trajectory visits multiple distinct low-loss basins; the
   pre-restart iterates are diverse (this is a pre-method observation about restart trajectories;
   the ensemble *result* is out of scope as it's the proposed method's eval / a follow-up).

## Derivation backbone (the discovery path) for reasoning.md
1. Start from the staircase pain. Why does step decay even work? Because a high LR explores broadly /
   escapes shallow regions, a low LR fine-tunes. The staircase is a crude two-phase explore-then-refine.
   But it's discontinuous and the drop points are arbitrary tuned numbers, and T must be fixed.
2. The restart idea from gradient-free + O'Donoghue: a momentum method's effectiveness is tied to a
   condition number that varies as you move; periodic restart of the momentum phase robustly recovers
   fast convergence without knowing the curvature. Want to import this to SGD-for-DL.
3. But literally resetting momentum/iterate is harsh and throws away learned velocity; and the proper
   restart test (O'Donoghue's f-increase / gradient-angle) needs denoised f/grad in the stochastic
   setting. Wall: per-batch f and grad are too noisy to drive a restart test reliably; would need
   epoch-averaged quantities.
4. So *emulate* a restart instead: don't reset state, just raise the LR back up. A warm restart =
   keep x_t (and momentum) but push eta back to eta_max. The amount of the LR increase controls how
   much old momentum is "used" vs overridden. This sidesteps the noisy restart-test problem and keeps
   the partial progress.
5. Now what shape between restarts? Need to go from eta_max down to a small eta_min smoothly (continuous,
   unlike step). Candidate shapes: linear (triangular, Smith) / exponential / polynomial / cosine.
   Derive why cosine:
   - Want eta to *start* near eta_max and stay relatively high early (keep exploring) — derivative of
     the schedule near the top should be ~0 (slow initial decay).
   - Want eta to *end* near eta_min smoothly with derivative ~0 at the bottom (gentle landing, fine
     convergence, and a clean restart point).
   - A function on [0,1] (s=T_cur/T_i) with f(0)=1, f(1)=0, f'(0)=f'(1)=0, monotone decreasing:
     the half-cosine 0.5*(1+cos(pi*s)) is exactly this. cos starts flat at 0, ends flat at pi.
     Linear has constant slope (no flat top/bottom); exponential is steep early, flat late (wrong way
     round — drops the exploratory LR too fast); polynomial (1-s)^p flattens at the bottom only.
     The half-cosine is the natural smooth "stay high, then accelerate down, then ease in" curve with
     zero slope at both ends. Derive 0.5*(1+cos(pi s)) by requiring those boundary conditions.
   - General affine map to [eta_min,eta_max]: eta = eta_min + (eta_max-eta_min)*0.5*(1+cos(pi*T_cur/T_i)).
6. Check endpoints: T_cur=0 -> cos0=1 -> eta_max (the restart kick). T_cur=T_i -> cos(pi)=-1 -> eta_min
   (the gentle bottom = snapshot/recommendation point).
7. T_cur is updated every batch, so it takes fractional values (0.1, 0.2,...) -> the per-batch curve is
   smooth, not per-epoch piecewise.
8. Restart period: fixed T_i=T_0 every cycle, OR (borrowing the CMA-ES "increase the budget per restart")
   geometrically grow it: T_i = T_0 * T_mult^i (e.g. T_0=1 or 10, T_mult=2 doubles each cycle). Why
   doubling: anytime performance — first restart comes quickly to get an early decent solution, then
   each cycle gets longer to refine, mirroring the small-lambda-then-double CMA-ES strategy.
9. Keep eta_max,eta_min fixed across restarts for simplicity (fewer hyperparams); note one *could*
   decay them per restart (forward-looking).
10. Recommendation/incumbent: warm restart temporarily worsens performance (the LR kick), so don't
    report the last x_t; report the end-of-run x_t at eta=eta_min (the bottom of each cosine). No
    separate validation set needed.
11. Single-run special case: with no restart (T_i=T = total budget) and eta_min=0 this is just
    eta_t = eta_max * 0.5*(1+cos(pi*T_cur/T)) — a smooth full-cosine decay to 0. This alone already
    beats / matches step decay (the paper finds T_0=200,Tmult=1 best on CIFAR-10). THIS is the
    MLS-Bench "cosine" baseline.
12. Land on code: get_lr / CosineAnnealingLR closed form (eta_min + (base-eta_min)*(1+cos(pi*t/T))/2),
    plus the restart bookkeeping (T_cur, T_i, T_mult) for CosineAnnealingWarmRestarts.

## Design-decision -> why table
- cosine half-wave shape: f(0)=1,f(1)=0,f'(0)=f'(1)=0 boundary conditions; flat top (keep exploring),
  flat bottom (fine convergence + clean restart). Alternatives: linear=constant slope no flats;
  exp=too steep early; poly=flat bottom only. -> cosine uniquely smooth at BOTH ends.
- warm restart (raise LR) instead of literal momentum/iterate reset (O'Donoghue): keeps partial
  progress (x_t, velocity); the noisy stochastic setting makes the proper restart-TEST unreliable
  per-batch, so a *scheduled* restart (raise eta) is the robust emulation; the size of the LR jump
  tunes how much old info survives.
- eta_min usually 0 (or tiny): anneal all the way down for a sharp final fit and a maximal-contrast
  restart kick.
- T_mult doubling: anytime performance — fast first solution, progressively longer refinement; inherited
  from CMA-ES increase-budget-per-restart.
- per-batch T_cur (fractional): smooth within-epoch curve; the staircase's discontinuity is gone.
- fixed eta_max/eta_min across restarts: minimize hyperparameters (just eta_max and T).
- recommend end-of-cycle iterate (eta=eta_min), not last iterate: restart kick worsens performance
  transiently; the cosine bottom is the good incumbent; removes need for a validation split.

## In-frame discipline notes
- NEVER name SGDR-as-paper / "the paper" / authors / arXiv id in any deliverable. May name the schedule
  ("cosine annealing", "warm restart") as the thing being built, mainly in answer.md.
- Cite ancestors freely: O'Donoghue & Candes 2012, Smith 2015, Loshchilov-Schoenauer-Sebag 2012,
  Hansen 2009, He 2015 (ResNet), Zagoruyko & Komodakis 2016 (WRN), Nesterov 1983, Fletcher-Reeves 1964,
  Powell 1977, Dauphin 2014.
- Pre-method scaffold (context.md): a generic per-epoch LR schedule slot — get_lr(epoch,...) returning a
  float, called by an existing SGD+momentum loop. ONE empty slot. Do NOT pre-name cosine/restart/T_cur.
- No proposed-method eval numbers (no 3.14%/16.21%/19.58% etc.). Step-decay milestones/settings (WRN
  60/120/160, ResNet 32k/48k) ARE pre-method facts -> background/baselines OK.
