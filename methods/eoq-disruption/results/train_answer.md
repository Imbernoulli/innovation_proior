The problem is lot-sizing when the supplier can suddenly stop shipping. Demand is constant at rate D, each order costs a fixed F plus a per-unit purchase cost a, holding one unit costs h per unit time, and every unit of demand that cannot be filled costs pi. The supplier alternates between an up state and a down state according to a two-state continuous-time Markov chain: it fails at rate lambda and recovers at rate psi. The classical EOQ balances ordering and holding under the hidden assumption that the supplier is always ready, so it can leave the firm with no stock exactly when the supplier is down. Numerically minimizing the exact renewal-reward cost is accurate, but it produces only a number, not a formula that can be embedded in larger models or used for comparative statics. What is needed is an order quantity that explicitly carries disruption risk and collapses back to the classical EOQ when lambda goes to zero.

The method is EOQ with Supply Disruptions, abbreviated EOQD. It treats the time between two consecutive successful shipments as a renewal cycle and applies the renewal-reward theorem: the long-run average cost per unit time is the expected cost incurred in one cycle divided by the expected length of that cycle. A cycle starts right after a shipment arrives, with inventory Q and the supplier up. Demand depletes the stock in Q/D time. The probability that the supplier has gone down by then is phi(Q/D) = lambda/(lambda+psi) * (1 - exp(-(lambda+psi) Q/D)). If it is down, the memoryless exponential down-time gives an expected extra wait of 1/psi. Therefore the expected cycle length is E[T] = Q/D + A0 (1 - exp(-(lambda+psi) Q/D)), where A0 = lambda/(psi(lambda+psi)).

The expected cost per cycle has three parts. Ordering cost is F + aQ. Holding cost is the area of the classical triangle, h Q^2/(2D). Shortage cost is pi D (E[T] - Q/D), which counts only the units actually lost while the firm waits at zero inventory. Dividing expected cost by expected cycle length gives the exact long-run cost rate I(Q) = pi D + (F + aQ + h Q^2/(2D) - pi Q)/E[T]. This cost is quasiconvex, so its minimum can be found with golden-section or bisection search.

For a closed form, EOQD drops the exponential term, which is small whenever the recovery rate psi is moderately large. Then E[T] is approximated by Q/D + A0, a constant disruption buffer added to the depletion time. The approximate cost becomes a ratio of a quadratic over an affine function of Q, and its first-order condition reduces to (h/(2D)) Q^2 + h A0 Q + ((a - pi) A0 D - F) = 0. The positive root is Q_hat = (-h A0 + sqrt(h^2 A0^2 + (2h/D)(F + (pi - a) A0 D))) / (h/D). When lambda is zero, A0 is zero and Q_hat simplifies to sqrt(2FD/h), the classical EOQ. When shortages are costly enough that pi - a exceeds h Q_E / D, the formula orders more than the EOQ; the excess Q_hat - Q_E is safety stock against supply unavailability. If shortages are cheap, it does not artificially inflate the order.

```python
import math

def cycle_buffer_constant(lam, psi):
    """A0 = lambda / (psi (lambda+psi)): disruption buffer in expected cycle time."""
    return lam / (psi * (lam + psi))

def expected_cycle_length(Q, D, lam, psi):
    A0 = cycle_buffer_constant(lam, psi)
    return Q / D + A0 * (1.0 - math.exp(-(lam + psi) * Q / D))

def cost_rate(Q, D, F, a, h, pi, lam, psi):
    """Exact long-run average cost per unit time via renewal-reward."""
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
    """Closed-form EOQD approximation; reduces to sqrt(2FD/h) when lambda=0."""
    A0 = cycle_buffer_constant(lam, psi)
    qa = h / (2.0 * D)
    qb = h * A0
    qc = (a - pi) * A0 * D - F
    disc = qb * qb - 4.0 * qa * qc
    return (-qb + math.sqrt(disc)) / (2.0 * qa)

def classical_eoq(D, F, h):
    """Harris EOQ."""
    return math.sqrt(2.0 * F * D / h)
```

The implementation provides both the exact quasiconvex cost and the closed-form EOQD approximation. Use the numerical minimizer when a precise value is enough, and use the approximation when an algebraic expression is needed. Both reduce cleanly to the classical EOQ in the absence of disruptions.
