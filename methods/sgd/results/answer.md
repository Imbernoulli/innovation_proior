# Stochastic Gradient Descent (SGD), distilled

SGD minimizes an expected loss `F(w) = E_z[Q(z, w)]` by following a *noisy* estimate of the
gradient instead of the exact one. Each step replaces the full-batch gradient `(1/n) sum_i grad
Q(z_i, w)` with the gradient on a single randomly drawn example (or a small mini-batch),

```
w_{t+1} = w_t - gamma_t * grad Q(z_t, w_t),
```

where `g_t = grad Q(z_t, w_t)` is an unbiased, noisy estimate of `grad F(w_t)`. It is the
direct application of the stochastic-approximation (Robbins-Monro) recursion for finding the
root of an unknown mean function, applied to `grad F = E[grad Q]` with target `0`. Cost is
`O(1)` per step (independent of `n`), first-order only, no matrices.

## Problem it solves

Minimizing an expected loss that can only be observed through noise — the data distribution is
unknown, or `n` is so large that an exact gradient (cost `n` per step) is unaffordable — using
cheap, `n`-independent first-order updates, with a guarantee that the iterates approach the
optimum despite per-step gradient noise.

## Key idea

The gradient of the expected loss is the expectation of the per-example gradient: `grad F(w) =
E_z[grad Q(z, w)]`. So minimizing `F` (finding `grad F = 0`) is a noisy root-finding problem,
and one noisy sample `grad Q(z_t, w_t)` is an unbiased draw of `grad F(w_t)`. Descending along
that single noisy sample is cheap and, in expectation, correct. The subtlety is the step size:
a fixed step converges only to a noise ball, while a diminishing step with the classical
summability conditions averages the noise away asymptotically.

## The step-size conditions (and why)

Decompose the iterate error into a deterministic part (memory of the start) and a random part
(accumulated gradient noise). The two require opposite things of the step sizes `gamma_t`:

- `sum_t gamma_t = infinity` — so the deterministic error `prod_t (1 - gamma_t) -> 0` and the
  method can travel arbitrarily far from a bad start (forget the initialization).
- `sum_t gamma_t^2 < infinity` — so the accumulated noise stays finite and shrinks (the steps
  decay fast enough that the variance term vanishes).

The canonical schedule satisfying both is `gamma_t ~ 1/t` (`sum 1/t = infinity`, `sum 1/t^2 =
pi^2/6 < infinity`); for mean estimation, `gamma_t = 1/t` makes the recursion *exactly* the
running sample mean. In practice start tame and decay into the `1/t` regime: `gamma_t = gamma_0
/ (1 + gamma_0 lambda t)`. Its asymptotic constant is `1/lambda`, so the strongly convex
`O(1/k)` induction requires choosing it with the strict slack `1/lambda > 1/(c mu)`.

## Convergence theory

**Robbins-Monro (root-finding form).** For an unknown monotone `M(x) = E[Y(x)]` observed
through bounded noisy `Y`, the recursion `x_{n+1} - x_n = a_n(alpha - y_n)` converges in
quadratic mean (hence in probability) to the root `theta` of `M(x) = alpha`, provided

```
sum a_n^2 < infinity   and   sum_{n>=2} a_n / (a_1 + ... + a_{n-1}) = infinity   (=> sum a_n = infinity).
```

The proof tracks `b_n = E(x_n - theta)^2`. Then `b_{n+1} - b_n = a_n^2 e_n - 2 a_n d_n`,
with drift `d_n = E[(x_n - theta)(M(x_n) - alpha)] >= 0` (monotonicity) and bounded spread
`0 <= e_n <= (C + |alpha|)^2`. Summing and using `b_n >= 0` gives `sum a_n d_n < infinity` and
a limit `b = lim b_n`. Lower-bounding `d_n >= kbar_n b_n` with `kbar_n = inf{(M(x)-alpha)/(x-
theta)}` over the reachable interval, and showing `sum a_n kbar_n = infinity`, forces `b = 0`.
The argument is distribution-free (no model for `M` or `H`). It generalizes least squares,
which instead assumes a linear form for `M` and fits in batch.

**Optimization form (smooth, strongly convex).** With `L`-Lipschitz gradient and curvature `c
> 0`, and gradient-noise second moment `E[||g_k||^2] <= M + M_G ||grad F(w_k)||^2`, the descent
inequality gives

```
E[F(w_{k+1}) | w_k] - F(w_k) <= -(mu - (1/2) gamma_k L M_G) gamma_k ||grad F(w_k)||^2 + (1/2) gamma_k^2 L M.
```

Two competing terms: signal `-gamma ||grad F||^2` and noise floor `+gamma^2 M`.

- **Constant step** `gamma <= mu/(L M_G)`: linear convergence to a *noise ball*,
  ```
  E[F(w_k) - F*] <= (gamma L M)/(2 c mu) + (1 - gamma c mu)^{k-1}(F(w_1) - F* - (gamma L M)/(2 c mu))
                --> (gamma L M)/(2 c mu).
  ```
  Fast, but only to a neighborhood of size proportional to `gamma`; smaller step = smaller ball
  but slower contraction. With no noise (`M = 0`) it recovers exact linear convergence to `F*`.
  This is the right reading of a *fixed* learning rate run to a plateau.
- **Diminishing step** `gamma_k = beta/(gamma + k)` with `beta > 1/(c mu)` (and `gamma_1`
  small enough): `E[F(w_k) - F*] <= nu/(gamma + k) = O(1/k)`, with `nu = max{beta^2 L M/(2(beta
  c mu - 1)), (gamma+1)(F(w_1)-F*)}`. True convergence to `F*`; the condition `beta > 1/(c mu)`
  is the sharp "don't shrink the steps too fast" requirement.

Per iteration SGD is slower than batch descent (which is linear, `~ rho^t`, vs SGD's `1/k`),
but per unit *compute* it is faster to a target expected risk in the large-scale regime,
because each step costs `1` rather than `n`. Scaling the step by curvature (stochastic
second-order) improves constants but not the noise-limited `1/k` tail — the variance term is
unchanged — so the way past the noise floor is averaging (mini-batches, decaying steps, or
iterate averaging).

## Working code

Filling the `step()` slot of the diagonal-net optimizer harness. Vanilla constant-step SGD is
the minimal form: subtract a fixed multiple of the gradient from each parameter vector and
return the updated `(u, v, state)` tuple.

```python
from typing import Any
import torch


def get_hyperparameters(dim: int, sparsity: int, delta: float) -> dict[str, Any]:
    """Single knob: the step size gamma."""
    return {"lr": 0.1}


def init_state(u: torch.Tensor, v: torch.Tensor,
               hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Bare SGD keeps no per-parameter state."""
    return {"t": 0}


def step(u: torch.Tensor, v: torch.Tensor,
         grad_u: torch.Tensor, grad_v: torch.Tensor,
         state: dict[str, Any], hyperparameters: dict[str, Any]
         ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """w_{t+1} = w_t - gamma * g_t (stochastic-approximation recursion, alpha=0)."""
    lr = float(hyperparameters["lr"])
    state["t"] = state.get("t", 0) + 1
    return u - lr * grad_u, v - lr * grad_v, state
```
