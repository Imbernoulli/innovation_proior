# EOQ with Supply Disruptions (EOQD)

## Problem

A firm buys one item from a single supplier and faces constant continuous demand at rate $D$. Ordering costs a fixed $F$ plus $a$ per unit; holding costs $h$ per unit per year; an unmet demand (lost sale) costs $\pi$ per unit. Unlike the classical lot-sizing model, the supplier is not always available: it alternates between an "up" (can ship) and a "down" (cannot ship) state, with up-times exponential at rate $\lambda$ (the **disruption rate**) and down-times exponential at rate $\psi$ (the **recovery rate**). Replenishment is otherwise instantaneous and the firm uses zero-inventory ordering. How large should each order $Q$ be, and what is the long-run average cost?

## Key idea

Because the firm may hit zero inventory while the supplier is down and then wait, the time between successful replenishments is a random variable. Treat each interval between consecutive shipments as a renewal cycle and apply the **renewal-reward theorem**: long-run average cost per unit time equals expected cost per cycle over expected cycle length. The expected cycle length is the classical depletion time $Q/D$ plus an expected wait for a disrupted supplier; this couples the cost to the two-state Markov supply process. The resulting exact cost rate is quasiconvex but transcendental (it contains $e^{-(\lambda+\psi)Q/D}$), so the optimum is found numerically. A tight closed-form approximation drops that exponential — negligible whenever the recovery rate $\psi$ is moderately large — which turns the expected cycle length into a constant buffer $Q/D + A_0$ and reduces the first-order condition to a quadratic in $Q$. The whole model collapses to the classical EOQ when $\lambda=0$; in the usual costly-stockout regime, the closed form expands the order quantity above EOQ and the difference is safety stock against supply unavailability.

## Algorithm

Let $A_0 = \dfrac{\lambda}{\psi(\lambda+\psi)}$.

**Supplier down-probability** (two-state CTMC started up):
$$ \phi(t) = \frac{\lambda}{\lambda+\psi}\left(1-e^{-(\lambda+\psi)t}\right). $$

**Expected cycle length** (depletion time plus expected wait $\tfrac1\psi$ taken with probability $\phi(Q/D)$):
$$ E[T] = \frac{Q}{D} + A_0\left(1-e^{-(\lambda+\psi)Q/D}\right). $$

**Exact long-run average cost** (renewal-reward; holding triangle $hQ^2/2D$, shortage $\pi D(E[T]-Q/D)$):
$$ I(Q) = \pi D + \frac{F + aQ + \dfrac{hQ^2}{2D} - \pi Q}{E[T]}. $$

$I(Q)$ is quasiconvex; minimize by golden-section or bisection to get the exact $Q^*$.

**Tight approximation.** For realistic $\psi$, $e^{-(\lambda+\psi)Q/D}\approx 0$, so $E[T]\approx Q/D + A_0$ and, in the costly-shortage regime, $I(Q)$ becomes a convex quadratic-over-affine ratio. Its first-order condition is
$$ \frac{h}{2D}\,Q^2 + hA_0\,Q + \big((a-\pi)A_0 D - F\big) = 0, $$
with positive root
$$ \hat Q = \frac{-hA_0 + \sqrt{h^2A_0^2 + \dfrac{2h}{D}\big(F+(\pi-a)A_0 D\big)}}{h/D}. $$

When $\lambda=0$ (so $A_0=0$), $\hat Q = \sqrt{2FD/h} = Q_E$, the classical EOQ. With $A_0>0$, the expansion above EOQ occurs when
$$ \pi-a > \frac{hQ_E}{D}; $$
then $\hat Q-Q_E$ is the safety stock against disruptions. If shortage is barely costly, the same formula can keep the order quantity near or below $Q_E$.

## Code

```python
import math

def cycle_buffer_constant(lam, psi):
    """A0 = lambda / (psi (lambda+psi)): expected wait-weighting the disruption
    adds to the cycle. Zero when lambda=0 (no disruptions)."""
    return lam / (psi * (lam + psi))

def expected_cycle_length(Q, D, lam, psi):
    """E[T] = Q/D + A0 (1 - e^{-(lambda+psi) Q/D}). Depletion time plus the
    expected extra wait when the supplier is down at reorder."""
    A0 = cycle_buffer_constant(lam, psi)
    return Q / D + A0 * (1.0 - math.exp(-(lam + psi) * Q / D))

def cost_rate(Q, D, F, a, h, pi, lam, psi):
    """Exact long-run average cost per unit time via renewal-reward:
    pi*D + (F + a Q + h Q^2/2D - pi Q) / E[T]. Quasiconvex in Q."""
    ET = expected_cycle_length(Q, D, lam, psi)
    return pi * D + (F + a * Q + h * Q * Q / (2.0 * D) - pi * Q) / ET

def optimize_exact(D, F, a, h, pi, lam, psi, lo=1e-9, hi=None, tol=1e-9):
    """Golden-section search; valid because cost_rate is single-troughed."""
    if hi is None:
        hi = 50.0 * math.sqrt(2.0 * F * D / h) + D
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
    fc = cost_rate(c, D, F, a, h, pi, lam, psi)
    fd = cost_rate(d, D, F, a, h, pi, lam, psi)
    while hi - lo > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo); fc = cost_rate(c, D, F, a, h, pi, lam, psi)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo); fd = cost_rate(d, D, F, a, h, pi, lam, psi)
    Q = 0.5 * (lo + hi)
    return Q, cost_rate(Q, D, F, a, h, pi, lam, psi)

def approximate_order_quantity(D, F, a, h, pi, lam, psi):
    """Tight closed form. Drop e^{-(lambda+psi)Q/D}: E[T] ~ Q/D + A0,
    so the first-order condition is the quadratic
        (h/2D) Q^2 + (h A0) Q + ((a - pi) A0 D - F) = 0.
    Positive root; reduces to sqrt(2FD/h) when lambda=0."""
    A0 = cycle_buffer_constant(lam, psi)
    qa = h / (2.0 * D)
    qb = h * A0
    qc = (a - pi) * A0 * D - F
    disc = qb * qb - 4.0 * qa * qc
    return (-qb + math.sqrt(disc)) / (2.0 * qa)

def classical_eoq(D, F, h):
    """Harris EOQ: sqrt(2 F D / h)."""
    return math.sqrt(2.0 * F * D / h)
```
