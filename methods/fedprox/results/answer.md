# FedProx, distilled

FedProx is a federated optimization framework that makes two lightweight changes to FedAvg to
handle heterogeneous device networks: (1) each client locally minimizes its empirical risk
plus a **proximal term** `(mu/2)||w - w^t||^2` anchoring the solution to the round's broadcast
model, and (2) clients are allowed to do **variable, inexact amounts of local work**
(`gamma`-inexact solves) and have their partial solutions aggregated instead of being dropped.
The server-side merge is unchanged from FedAvg (sample-weighted averaging). FedAvg is the
special case `mu = 0` with an SGD solver and a fixed amount of local work.

## Problem it solves

Minimize `f(w) = sum_k p_k F_k(w)`, `p_k = n_k/n`, over a network of devices that (a) hold
*non-identically distributed* (non-IID) data, so the local risks `F_k` and their minimizers
differ from the global objective (statistical heterogeneity), and (b) have very different
compute/network/battery, so they cannot all perform the same amount of local work (systems
heterogeneity). Communication is the bottleneck, only a fraction of devices participate per
round, and `F_k` may be non-convex. The goal is robust convergence under both heterogeneities,
with a convergence guarantee.

## Key idea

Under non-IID data, more local SGD epochs drive each device toward its *own* local optimum,
away from the global one — "client drift" — which makes averaging the returned models
unstable or divergent, while the only baseline control (number of local epochs `E`) is coarse
and fixed per round. FedProx instead caps the drift directly by changing the *local objective*:
device `k` approximately minimizes

```
h_k(w; w^t) = F_k(w) + (mu/2) ||w - w^t||^2
```

a quadratic spring anchored at the broadcast model `w^t`. Its gradient is `∇F_k(w) + mu(w - w^t)`,
so each local SGD step becomes `w <- w - eta(∇F_k(w) + mu(w - w^t))` — ordinary local SGD plus
a restoring force toward `w^t`. The spring:

- **Tethers client drift** regardless of solver or number of steps, so local solutions stay in
  a shared basin and their average is meaningful. `mu = 0` recovers FedAvg exactly.
- **Convexifies the local subproblem.** If `∇^2 F_k ≥ -L_ I`, then `∇^2 h_k ≥ (mu - L_) I`, so
  with `bar_mu := mu - L_ > 0` the subproblem is `bar_mu`-strongly convex even for non-convex
  `F_k`. This gives the displacement bound `||argmin h_k - w^t|| ≤ (1/bar_mu)||∇F_k(w^t)||` that
  the whole convergence proof rests on.

For systems heterogeneity, FedProx does not mandate a fixed `E` or drop stragglers. It measures
local work by **`gamma`-inexactness**: `w_k` is a `gamma_k^t`-inexact solution of `h_k` if

```
||∇h_k(w_k; w^t)|| ≤ gamma_k^t ||∇h_k(w^t; w^t)|| = gamma_k^t ||∇F_k(w^t)||,   gamma_k^t in [0,1],
```

with `gamma = 0` exact and `gamma = 1` no work; the number of local iterations is a proxy. Each
device does what it can, returns a partial `w_k`, and it is aggregated — the inexactness is
paid for in the bound, not by exclusion. Unlike DANE/AIDE, there is no gradient-correction term
(it would need the full global gradient `∇f`, impractical under low participation).

## Defaults and why

The canonical implementation uses `mu = 0.01` as its optimizer default, with a suggested range
of about `0.001`–`0.1` and an adaptive heuristic that nudges `mu` up by `0.1` when the loss
rises and down by `0.1` when it falls. Theory wants `mu > L_` for convexity and *larger* `mu`
for higher target accuracy (convex case: `mu ≈ 6LB^2`); in practice, `mu` is a stability knob
that should be tuned with the local epoch count and the degree of heterogeneity.

## Algorithm

```
Input: K (devices/round), T (rounds), mu, gamma, w^0, N, p_k = n_k/n
for t = 0 ... T-1:
    server selects a subset S_t of K devices (device k w.p. p_k)
    server broadcasts w^t to all k in S_t
    each k in S_t finds w_k^{t+1} ≈ argmin_w  h_k(w; w^t) = F_k(w) + (mu/2)||w - w^t||^2
        (a gamma_k^t-inexact solution; any local solver; variable amount of work)
    each k in S_t returns w_k^{t+1}
    server aggregates:  w^{t+1} = sum_{k in S_t} (n_k / sum_{j in S_t} n_j) w_k^{t+1}
                        (the analysis also uses 1/K averaging under p_k sampling)
```

## Convergence (non-convex, non-IID, partial participation)

Assume each `F_k` is `L`-smooth with `∇^2 F_k ≥ -L_ I`, `bar_mu = mu - L_ > 0`, and `B`-local
dissimilarity `E_k||∇F_k(w)||^2 ≤ B^2 ||∇f(w)||^2` (with masses `p_k`; `B = 1` when all `F_k`
coincide, `B ≥ 1` measures heterogeneity). Then one round gives an expected decrease

```
E_{S_t}[f(w^{t+1})] ≤ f(w^t) - rho ||∇f(w^t)||^2,
rho = 1/mu - gamma B/mu - B(1+gamma)sqrt(2)/(bar_mu sqrt(K)) - LB(1+gamma)/(bar_mu mu)
      - L(1+gamma)^2 B^2/(2 bar_mu^2) - L B^2 (1+gamma)^2/(bar_mu^2 K)(2 sqrt(2K) + 2).
```

The full displayed `rho` must be positive. Two necessary qualitative constraints are
`gamma B < 1` and `B/sqrt(K) < 1`: sloppy local solves require a more homogeneous network, and
heterogeneity requires enough sampled devices per round. Telescoping the decrease gives, after
`T = O(Delta/(rho epsilon))` rounds (`Delta = f(w^0) - f^*`),
`(1/T) sum_t E||∇f(w^t)||^2 ≤ epsilon`.

**Proof sketch.** (1) `gamma`-inexactness ⇒ residual `e_k = ∇h_k(w_k^{t+1})`, `||e_k|| ≤
gamma||∇F_k(w^t)||`. (2) `bar_mu`-strong convexity ⇒ `||w_k^{t+1} - w^t|| ≤ (1+gamma)/bar_mu
||∇F_k(w^t)||`; Jensen + dissimilarity ⇒ `||bar_w^{t+1} - w^t|| ≤ B(1+gamma)/bar_mu ||∇f||`,
where `bar_w^{t+1} = E_k[w_k^{t+1}]`. (3) Writing `bar_w^{t+1} - w^t = -(1/mu)(∇f + M_{t+1})`,
bound `||M_{t+1}|| ≤ (L(1+gamma)/bar_mu + gamma)B||∇f||`; the `L`-smooth descent lemma at
`bar_w^{t+1}` gives the per-round decrease for full participation. (4) The `K`-device sample
mean has `E_{S_t}||w^{t+1} - bar_w^{t+1}||^2 ≤ (2B^2/K)(1+gamma)^2/bar_mu^2 ||∇f||^2`; local
Lipschitz continuity pays for the sampling gap. Combining (3)+(4) yields `rho`.

**Convex case / relation to SGD.** With `F_k` convex (`bar_mu = mu`), `gamma = 0`, and
`1 << B ≤ 0.5 sqrt(K)`, choosing `mu ≈ 6LB^2` gives `rho ≈ 1/(24LB^2)`. Under bounded variance
`E_k||∇F_k - ∇f||^2 ≤ sigma^2` one has `B_epsilon ≤ sqrt(1 + sigma^2/epsilon)`, and the round
count becomes `O(LDelta/epsilon + LDelta sigma^2/epsilon^2)` — the same as SGD. FedProx matches
(does not beat) distributed SGD asymptotically; the analysis gives sufficient conditions for
convergence and characterizes how heterogeneity hurts, not a superiority result. The practical
gain is stability/robustness under heterogeneity.

**Variable work.** Replacing `gamma` by `gamma^t = max_{k in S_t} gamma_k^t` extends the bound
to per-device, per-round inexactness (since `E_k[(1+gamma_k^t)||∇F_k||] ≤ (1 + max gamma_k^t)
E_k||∇F_k||`), so stragglers contribute partial solutions instead of being dropped.

## Working code

The local objective is `F_k + (mu/2)||w - w^t||^2`, realized as
`var <- var - lr * (grad + mu * (var - vstar))`, with `vstar` holding the broadcast model. The
server merge is the FedAvg sample-weighted mean.

```python
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.training import optimizer
from .fedbase import BaseFedarated


class PerturbedGradientDescent(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="PGD"):
        super(PerturbedGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu
        self._lr_t = None
        self._mu_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "vstar", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")
        var_update = state_ops.assign_sub(
            var, lr_t * (grad + mu_t * (var - vstar))
        )
        return control_flow_ops.group(var_update)

    def set_params(self, global_params, client):
        with client.graph.as_default():
            for variable, value in zip(tf.trainable_variables(), global_params):
                self.get_slot(variable, "vstar").load(value, client.sess)


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        self.inner_opt = PerturbedGradientDescent(
            params["learning_rate"], params["mu"]
        )
        super(Server, self).__init__(params, learner, dataset)

    def train_round(self, round_num):
        _, selected_clients = self.select_clients(
            round_num, num_clients=self.clients_per_round
        )
        np.random.seed(round_num)
        active_clients = np.random.choice(
            selected_clients,
            round(self.clients_per_round * (1 - self.drop_percent)),
            replace=False,
        )
        self.inner_opt.set_params(self.latest_model, self.client_model)

        client_solutions = []
        for client in selected_clients.tolist():
            client.set_params(self.latest_model)
            if client in active_clients:
                solution, stats = client.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size
                )
            else:
                partial_epochs = np.random.randint(low=1, high=self.num_epochs)
                solution, stats = client.solve_inner(
                    num_epochs=partial_epochs, batch_size=self.batch_size
                )
            client_solutions.append(solution)
            self.metrics.update(rnd=round_num, cid=client.id, stats=stats)

        self.latest_model = self.aggregate(client_solutions)
        self.client_model.set_params(self.latest_model)

    def aggregate(self, weighted_solutions):
        total_weight = 0.0
        base = [0] * len(weighted_solutions[0][1])
        for weight, solution in weighted_solutions:
            total_weight += weight
            for i, value in enumerate(solution):
                base[i] += weight * value.astype(np.float64)
        return [value / total_weight for value in base]
```

This is FedProx as implemented: the perturbed-gradient step uses the frozen global model slot
`vstar`, and server aggregation remains the sample-weighted mean.
