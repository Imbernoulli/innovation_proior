# Context: making the gradient small in stochastic convex-concave saddle problems

## Research question

We want to solve smooth minimax problems
```
min_x max_y  f(x, y) = E_xi[ f(x, y; xi) ],
```
where `f` is `L`-smooth and we only ever see it through a stochastic first-order oracle: querying the point
`z = (x, y)` returns `F(z; xi) = F(z) + (noise)`, an unbiased estimate of the saddle gradient operator
```
F(z) = [  grad_x f(x, y) ;  - grad_y f(x, y) ]
```
with `E[F(z; xi)] = F(z)` and `E|| F(z; xi) - F(z) ||^2 <= sigma^2`. The success criterion we care about is a
**near-stationary point**: a `z` with `|| F(z) || <= eps`, i.e. the gradient (operator) norm is small.

This is a different, and harder, target than the classical one. For convex-concave problems the textbook quality
measure is the duality gap `max_{y'} f(x, y') - min_{x'} f(x', y)`, and there are methods that drive it to zero
optimally. But the duality gap is awkward: on an unbounded domain it can be infinite at every point except the
saddle (the bilinear `f(x,y) = x*y` has gap `+inf` for any `(x,y) != (0,0)`), and it is generally hard to measure.
The gradient-operator norm is always well defined, is cheap to read off, and keeps making sense without any
convex-concave assumption. The pain point is that an algorithm tuned to shrink one of these quantities is not
automatically good at shrinking the other — small suboptimality in value (or gap) does not imply small gradient.
So the goal is concrete: out of a stream of *noisy* saddle-gradient queries, produce a single iterate whose true
operator norm `|| F(z) ||` is small, on convex-concave (monotone) problems where naive methods either diverge or
stall at a noise floor.

## Background

The right abstraction is **monotone operators / variational inequalities** (Rockafellar 1970). A smooth
convex-concave `f` has a *monotone* gradient operator,
```
( F(z) - F(z') )^T ( z - z' ) >= 0      for all z, z',
```
and if `f` is `lambda`-strongly-convex-`lambda`-strongly-concave the operator is `lambda`-*strongly* monotone,
```
( F(z) - F(z') )^T ( z - z' ) >= lambda || z - z' ||^2.
```
Strong monotonicity is the property that buys contraction toward the unique solution `z*` (where `F(z*) = 0`);
plain monotonicity does not.

**The diagnostic that organizes everything: simultaneous gradient descent-ascent fails on saddle problems.**
The obvious method — descend in `x`, ascend in `y` at the same time, `z_{t+1} = z_t - eta F(z_t)` — diverges even
on the simplest convex-concave instance. Take `f(x,y) = x*y`, so `F(z) = (y, -x)`. Then `z^T F(z) = x*y + y*(-x) = 0`:
the update direction is exactly orthogonal to the vector pointing at the saddle, and
```
|| z_{t+1} ||^2 = || z_t - eta F(z_t) ||^2 = || z_t ||^2 - 2 eta z_t^T F(z_t) + eta^2 || F(z_t) ||^2
              = || z_t ||^2 ( 1 + eta^2 ),
```
so the iterates spiral strictly outward for any step size and any nonzero start. This rotational behavior — not a
descent failure but a geometry failure — is the phenomenon every saddle-point method has to defeat.

**Stochastic queries add a second obstacle: a noise floor.** Even for a method that contracts in the noiseless
case, replacing `F(z)` with an unbiased noisy `F(z; xi)` and using a *fixed* step size leaves an irreducible
residual: the iterate converges only to a neighborhood of `z*` whose radius scales with the per-query variance
`sigma^2` and the step size. Driving the last iterate all the way to `z*` (not just its average, and not just to a
ball around it) needs something more than a constant-step run of the naive method.

**The value-vs-gradient distinction.** It is documented (Nesterov 2012, "How to make the gradients small";
Foster, Sekhari, Shamir, Srebro, Sridharan & Woodworth, COLT 2019) that a method which is optimal for reducing
function value (or duality gap) can be far from optimal for reducing the gradient norm, and that these two
complexities genuinely differ in the stochastic setting. Existing convex-minimization analyses can exploit
strong convexity and objective-value comparisons; Allen-Zhu (2018) developed a stochastic convex scheme in that
line. Those arguments do not immediately apply to minimax problems, which are governed by monotone operators and
saddle-gradient norms rather than a scalar value ordering. The open difficulty is to make the last-iterate
operator norm small under stochastic noise when the strong-monotone contraction used in the clean analysis is
absent.

## Baselines

**Gradient descent-ascent (GDA).** `z_{t+1} = z_t - eta F(z_t)`. The natural first thing to try. As shown above it
*diverges* on convex-concave saddle problems with a rotational operator (bilinear), and even when it converges
(strongly-convex-strongly-concave with small step) it is fragile. **Gap:** the single-gradient step cannot cancel
the rotational component of `F`; it has no mechanism to look past the local linearization.

**The extragradient method (Korpelevich 1976, "The extragradient method for finding saddle points and other
problems," Matecon 12:747-756).** Repair GDA with a *lookahead*: take a trial half-step, re-evaluate the operator
at the trial point, and step from the *original* point with that re-evaluated gradient,
```
z_{t+1/2} = z_t - eta F(z_t),
z_{t+1}   = z_t - eta F(z_{t+1/2}).
```
The second gradient `F(z_{t+1/2})` carries the information the single GDA gradient missed; on the bilinear problem
the lookahead produces a component that pulls toward the saddle instead of orthogonal to it, so the method
contracts where GDA expands. Extragradient can be read as a first-order approximation of the (implicit) proximal-
point step `z_{t+1} = z_t - eta F(z_{t+1})` (Mokhtari, Ozdaglar & Pattathil, AISTATS 2020), which contracts for
any monotone `F`; this is why it converges for monotone + Lipschitz operators and converges *linearly* in the
strongly-monotone and bilinear cases. **Gap:** on a *merely* monotone (not strongly monotone) operator the
guarantee is an `O(1/k)` rate on the duality gap of the *averaged* iterate — it controls the gap, not the
last-iterate gradient norm, and there is no contraction to lean on for `|| F ||`.

**Stochastic extragradient (SEG) (Juditsky, Nemirovski & Tauvel 2011; Mishchenko, Kovalev, Komodakis &
Richtarik, AISTATS 2020, "Revisiting Stochastic Extragradient"; Gorbunov, Berard, Gidel & Loizou, AISTATS 2022,
"Stochastic Extragradient: General Analysis and Improved Rates").** The same two-step scheme with the exact
operator replaced by independent unbiased noisy queries `F(z; xi_i)`, `F(z; xi_j)`. For a `lambda`-strongly-
monotone, `L`-Lipschitz operator with step `eta < 1/(4L)`, the per-step descent inequality takes the form
```
lambda E|| z_{t+1/2} - z* ||^2 <= (1/eta) E[ || z_t - z* ||^2 - || z_{t+1} - z* ||^2 ] + 16 eta sigma^2,
```
which telescopes to `E|| zbar - z* ||^2 <= || z_0 - z* ||^2 / (lambda eta T) + 16 eta sigma^2 / lambda`. **Gaps:**
(1) the bound needs *strong* monotonicity — on a plain convex-concave operator `lambda = 0` and the whole
contraction collapses, so SEG run directly on `F` has no last-iterate gradient-norm guarantee; (2) with the fixed
step the statistical term `16 eta sigma^2 / lambda` does not vanish with `T`, so SEG halts in a noise ball around
`z*` rather than at it.

**Anchored / Halpern-type extragradient with a diminishing anchor (Yoon & Ryu, ICML 2021, "Accelerated algorithms
for smooth convex-concave minimax problems with O(1/k^2) rate on squared gradient norm"; stochastic variant: Lee
& Kim, NeurIPS 2021).** Add to each extragradient step a pull toward the initial point `z_0` whose weight is a
*decreasing* schedule `~ 1/(k+2)`,
```
z_{t+1/2} = z_t - eta F(z_t) + beta_k ( z_0 - z_t ),   beta_k = 1/(k+2),
```
which yields the optimal `O(1/k)` deterministic rate on `|| F ||`. **Gap:** the deterministic anchoring schedule
does not transfer cleanly to the stochastic setting — its direct stochastic extension is observed to be
inefficient, with the variance interacting badly with the diminishing anchor weight.

## Evaluation settings

The natural yardsticks for a gradient-norm saddle method, all pre-existing:

- **Bilinear instance** `f(x,y) = x^T y` on `R^d x R^d`. The canonical pathology: GDA diverges, duality gap is
  ill-defined off the saddle, and first-order methods cycle. Here `F(z) = (y, -x)` so `|| F(z) || = || z ||`
  exactly, and the gradient norm is read off the iterate. A small fixed scalar instance (`z_0 = (10, 10)`) with
  additive Gaussian update noise of standard deviation `sigma` makes the rotational + noise interaction visible.
- **The `(delta, nu)` worst-case convex-concave instance** (the smooth-convex-minimization hard function of
  Drori & Teboulle, adapted by Yoon & Ryu),
  `f_{delta,nu}(x,y) = (1-delta) g_nu(x) + delta x^T y - (1-delta) g_nu(y)`, with `g_nu` the Huber-type function
  (quadratic for `|u| < nu`, linear with slope `nu` beyond), `delta = 1e-2`, `nu = 5e-5`, dimension `d = 100`.
  A structured, harder convex-concave test where the gradient norm `|| F(z) ||` is reported directly.
- **Protocol.** Additive Gaussian update noise `xi ~ N(0, sigma^2 I)` injected into each oracle query; a fixed
  number of outer iterations per problem; each iteration issues two stochastic operator queries (count SFO calls
  in pairs); the recorded quantity is the (true) gradient norm vs. iteration, with the per-problem starting point
  and RNG seed fixed. Identical noise model and iteration budget across the methods being compared.

## Code framework

The harness fixes the problem, the stochastic oracle, the noise model, the iteration count, the initialization,
and the metric; the only thing to fill in is the iterative update. The oracle exposes the deterministic saddle
gradient `oracle.grad(z) = F(z)` and a fresh additive-noise draw `oracle.noise() ~ N(0, sigma^2 I)`, so an update
can mirror its intended math directly. Three slots are open: how to set up per-run state, how to advance one
iteration, and what constants to use per problem. The update rule itself — the core of the method — is the empty
slot.

```python
from __future__ import annotations
from typing import Any
import numpy as np
from fixed_benchmark import (
    ProblemSpec, StepOutput, StochasticOracle,
    as_vector, make_step_output, run_cli,
)


def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    # Must preserve the provided starting point in state["z"].
    z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    # TODO: any per-run state the update rule we will design needs.
    return {"z": z0}


def step(
    state: dict[str, Any],
    oracle: StochasticOracle,        # oracle.grad(z) = F(z);  oracle.noise() ~ N(0, sigma^2 I)
    problem: ProblemSpec,
    hyperparameters: dict[str, Any],
    max_sfo_calls: int,
) -> StepOutput:
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    # TODO: one iteration of the saddle-point update we will design.
    #       Use oracle.grad / oracle.noise to advance z; return the next state,
    #       the iterate to measure the gradient norm at, and the number of
    #       stochastic operator queries this iteration spent.
    raise NotImplementedError


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    # TODO: the per-problem constants the method needs.
    raise NotImplementedError


if __name__ == "__main__":
    run_cli(init_state=init_state, step=step, get_hyperparameters=get_hyperparameters)
```
