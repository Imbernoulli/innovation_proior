# The Metropolis-Hastings Algorithm

## Method

For a target density `pi(x)=f(x)/K` known up to an unknown normalizer `K`, choose a proposal density `q(x,y)`. From current state `x`:

1. Propose `y ~ q(x,.)`.
2. Compute

   `R(x,y) = f(y) q(y,x) / (f(x) q(x,y))`.

3. Accept `y` with probability

   `alpha(x,y) = min(1, R(x,y))`.

4. If the proposal is rejected, keep `x` as the next state and count it again in the average.

The transition kernel is

`P(x,dy)=q(x,y) alpha(x,y) dy + r(x) delta_x(dy)`,

where `r(x)=1-int q(x,y) alpha(x,y) dy` is the rejected/staying mass.

## Why It Works

The acceptance rule is chosen to enforce detailed balance:

`pi(x) q(x,y) alpha(x,y) = pi(y) q(y,x) alpha(y,x)`.

If the proposal sends too much target-weighted flow from `x` to `y`, the accept/reject filter throttles exactly that direction. Since `pi=f/K`, the unknown `K` cancels from the ratio. Detailed balance implies stationarity by summing over source states:

`sum_i pi_i P_ij = sum_i pi_j P_ji = pi_j`.

In continuous space, the same proof uses the stay-put atom: the moving part contributes `(1-r(y))pi(y)` and rejected mass contributes `r(y)pi(y)`, so one transition preserves `pi`.

## Special Cases

- Symmetric proposal: if `q(x,y)=q(y,x)`, then `alpha=min(1, f(y)/f(x))`. For `f(x)=exp(-E(x)/kT)`, this is `min(1, exp(-Delta E/kT))`.
- Asymmetric proposal: the factor `q(y,x)/q(x,y)` corrects proposal bias.
- Barker filter: `R/(1+R)` is also reversible, but accepts less often. Peskun's ordering shows the maximal `min(1,R)` choice is variance-preferable for a fixed proposal among reversible kernels.

## Code Artifact

```python
import numpy as np

def mh_step(x, log_f, proposal, rng):
    """One Metropolis-Hastings transition.

    proposal(x, rng) returns (y, log_q_xy, log_q_yx).
    log_f is unnormalized; additive constants cancel.
    """
    y, log_q_xy, log_q_yx = proposal(x, rng)
    log_R = (log_f(y) - log_f(x)) + (log_q_yx - log_q_xy)
    if log_R >= 0.0 or np.log(rng.uniform()) < log_R:
        return y
    return x

def run_chain(x0, log_f, proposal, n_steps, burn_in, rng):
    x = x0
    kept = []
    for t in range(n_steps):
        x = mh_step(x, log_f, proposal, rng)
        if t >= burn_in:
            kept.append(np.array(x, copy=True))
    return kept
```

## Practical Conditions

Detailed balance gives invariance, not automatic fast exploration. The proposal support must make the target support reachable, the chain must avoid periodic trapping, and the proposal scale must be tuned: too large gives near-total rejection; too small gives slow diffusion. Rejected states remain part of the sample path.
