# AdaDelta — synthesis (grounded in arXiv 1212.5701 googleTR12.tex + pytorch torch/optim/adadelta.py)

## Verified arXiv id: 1212.5701 (Zeiler 2012, ICASSP-style tech report)
Canonical impl: PyTorch torch.optim.Adadelta (code/adadelta_pytorch.py). Defaults lr=1.0, rho=0.9, eps=1e-6.

## Pain point / research question
Choosing the learning rate eta for SGD is "more art than science": too high diverges, too low slow. Want a method that needs NO manual global learning rate, per-dimension, uses only first-order info, ~trivial overhead over SGD, robust to noise/architecture/hyperparams.

## Ancestors (all in the paper)
- SGD: Delta x_t = -eta g_t. One global eta, must be tuned.
- Momentum (autoencoder-nn ref): Delta x_t = rho Delta x_{t-1} - eta g_t. Accelerates consistent directions, damps sign-changing ones (per-dim effect), but still has eta.
- AdaGrad (Duchi 2011): Delta x_t = -eta / sqrt(sum_{tau=1..t} g_tau^2) * g_t. Per-dim dynamic rate = eta / L2-norm of ALL past grads. Large grads -> small rate, small grads -> large rate (evens out progress across dims/layers). Acts like annealing. Two drawbacks: (1) denominator accumulates from t=1 forever -> rate decays monotonically to ZERO, training stalls; (2) sensitive to initial gradient magnitude and to the global eta (large init grads -> permanently small rate).
- Newton: Delta x = H^{-1} g; optimal step for quadratics; H too big to compute.
- Becker & LeCun 1988: diagonal-Hessian approx, Delta x = -1/(|diag(H)|+mu) g. Costs extra fwd/back pass. |.| ensures descent; mu conditions small-curvature regions.
- Schaul et al. 2012: Delta x = -1/|diag(H)| * E[g_{t-w:t}]^2 / E[g^2_{t-w:t}] * g_t. Combines diag Hessian with AdaGrad-like windowed terms; needs the diagonal Hessian (extra cost) and a window heuristic. AdaDelta derived independently from AdaGrad, noticed similarity after.

## Idea 1: accumulate over a window (fix AdaGrad's decay-to-zero)
Replace the from-t=1 sum with a FIXED window w of recent squared gradients so the denominator can't grow unboundedly and stays a LOCAL estimate -> learning keeps progressing. Storing w grads is wasteful, so implement as an exponentially decaying average:
  E[g^2]_t = rho E[g^2]_{t-1} + (1-rho) g_t^2.
Since we need sqrt in the update, define RMS[g]_t = sqrt(E[g^2]_t + eps) (eps conditions denom, as Becker-LeCun).
Idea-1 update: Delta x_t = -eta / RMS[g]_t * g_t.   (still has a global eta!)

## Idea 2: correct the UNITS (remove the global eta) -- the key insight
Units mismatch: a parameter update should have the same units as the parameter x.
- SGD/Momentum: units(Delta x) ~ units(g) = units(df/dx) ~ 1/units(x)  (f unitless). WRONG units.
- AdaGrad and Idea-1: ratio of gradient quantities -> unitless. WRONG units.
- Newton: Delta x ~ H^{-1} g ~ (df/dx)/(d2f/dx2) ~ units(x). CORRECT units.
Rearrange Newton (diagonal H): Delta x = (df/dx)/(d2f/dx2)  =>  1/(d2f/dx2) = Delta x / (df/dx).
So the missing numerator unit-fixer is a quantity with units of Delta x. The denominator RMS[g] already supplies the (df/dx) part; put a measure of Delta x in the numerator. Delta x_t (current) is unknown (it's what we're computing), so assume locally-smooth curvature and approximate it by the RMS of PAST updates:
  Delta x_t = - RMS[Delta x]_{t-1} / RMS[g]_t * g_t.
RMS[Delta x]_{t-1} = sqrt(E[Delta x^2]_{t-1} + eps), with E[Delta x^2]_t = rho E[Delta x^2]_{t-1} + (1-rho) Delta x_t^2.
The eps in numerator: starts iter 1 (where Delta x_0 = 0) and keeps progress going when past updates get small.
NO global learning rate. The ratio RMS[Delta x]/RMS[g] ~ Delta x / g ~ 1/(d2f/dx2) = inverse diagonal-Hessian approx, always positive (RMS >=0) -> always descends -g_t.

## Final algorithm (Algorithm 1, exact)
init E[g^2]_0 = 0, E[Delta x^2]_0 = 0
for t=1..T:
  g_t = grad
  E[g^2]_t = rho E[g^2]_{t-1} + (1-rho) g_t^2
  Delta x_t = - RMS[Delta x]_{t-1} / RMS[g]_t * g_t        # RMS[.] = sqrt(. + eps)
  E[Delta x^2]_t = rho E[Delta x^2]_{t-1} + (1-rho) Delta x_t^2
  x_{t+1} = x_t + Delta x_t

## Properties / why it works (the "from each method" recap)
- Descent dir -g_t always followed (like SGD).
- Numerator RMS[Delta x] = acceleration/accumulation of past updates over a window (like momentum).
- Denominator RMS[g] = per-dim squared-gradient info evens out progress across dims (like AdaGrad) but WINDOWED so no decay-to-zero.
- Ratio ~ inverse diagonal Hessian (like Schaul/Becker-LeCun) but only ONE gradient eval/iter, no explicit Hessian.
- The numerator RMS[Delta x]_{t-1} LAGS denominator by 1 step (recurrence): side effect = robust to large sudden gradients (they spike the denominator, shrinking the effective rate THIS step, before the numerator reacts).
- Effective learning rate (all terms except g_t) larger for lower layers -> compensates vanishing gradients in backprop. Near end of training step sizes -> 1 (eps dominates as g and Delta x shrink), and updates -> 0 smoothly = implicit annealing.

## Design decisions -> why
- Windowed (EMA) accumulation of g^2 not full sum: prevents AdaGrad's monotone decay to zero; local estimate keeps learning.
- EMA implementation (rho, 1-rho) not literal window of w: O(1) memory, one extra buffer.
- RMS (sqrt of mean square + eps): need sqrt for the denominator magnitude; eps conditions denom (Becker-LeCun mu role).
- Numerator RMS[Delta x]: enforces correct units (parameter units), which is what lets you DROP the global eta; = inverse-Hessian-like scaling.
- Approximate unknown Delta x_t by RMS of PAST Delta x (lag 1): locally-smooth-curvature assumption; gives the large-gradient robustness side effect.
- eps in numerator too: bootstraps t=1 (Delta x_0=0) and keeps step from dying when past updates tiny.
- rho ~ 0.9-0.95, eps ~ 1e-6: shown insensitive (Table) -> the no-tuning claim.

## Code (PyTorch) mapping
state: square_avg = E[g^2], acc_delta = E[Delta x^2].
square_avg.mul_(rho).addcmul_(grad, grad, 1-rho)            # E[g^2]_t
std = square_avg.add(eps).sqrt_()                           # RMS[g]_t = sqrt(E[g^2]+eps)
delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)     # RMS[dx]_{t-1}/RMS[g]_t * g_t
p.data.add_(delta, alpha=-lr)                               # x += -lr*delta (lr=1.0 default = pure AdaDelta)
acc_delta.mul_(rho).addcmul_(delta, delta, 1-rho)           # E[dx^2]_t
Note PyTorch keeps an lr (default 1.0) as an optional outer scale; pure AdaDelta is lr=1.
