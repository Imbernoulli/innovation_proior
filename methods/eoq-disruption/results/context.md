# Context: lot-sizing when the supplier can vanish

## Research question

A firm buys a single item from one supplier, faces steady customer demand, and must decide how much to order each time it replenishes. The textbook answer is the economic order quantity. But that answer is built on an assumption that quietly fails in practice: that whenever the firm wants to order, the supplier is there to fill it instantly. Real suppliers are knocked offline — by fires, strikes, floods, equipment failure, port closures — for stretches of random length, during which they can ship nothing. A firm that sized its orders as if the supplier were always available will, sooner or later, run its inventory to zero exactly while the supplier is down, and then it loses sales (or backlogs them) until the supplier recovers.

The precise problem: given a supplier that alternates between an "up" state (can ship) and a "down" state (cannot), how large should each order be? The order quantity now trades off three things instead of two — ordering cost, holding cost, and the cost of being caught empty-handed during a disruption — and when unmet demand is expensive the natural guess is that the firm should carry *more* than the classical order quantity, holding the extra as a buffer against the supplier going dark. A solution must (i) quantify the long-run average cost as a function of the order size under random supply availability, and (ii) produce an order quantity that is implementable, ideally in a form as simple and embeddable as the classical formula.

## Background

**The classical lot-sizing model (Harris 1913).** Demand is deterministic and continuous at rate $D$. Each order costs a fixed $F$ (independent of size) plus a per-unit cost $a$; holding one unit for a year costs $h$. With instantaneous, always-available replenishment, the optimal policy is zero-inventory ordering (ZIO): place an order only when on-hand reaches zero. Inventory then sawtooths between $Q$ and $0$, the average on-hand is $Q/2$, and the long-run cost per unit time is

$$ \frac{FD}{Q} + aD + \frac{hQ}{2}. $$

Minimizing over $Q$ gives the economic order quantity $Q_E = \sqrt{2FD/h}$. The square root is the signature: ordering cost falls like $1/Q$, holding rises linearly, and they balance at $\sqrt{2FD/h}$. This is the object every disruption model perturbs.

**Renewal-reward as the accounting tool.** When a system regenerates — returns to a fresh, memoryless starting state at random epochs called renewals — the long-run average reward (or cost) per unit time equals the expected reward earned in one cycle divided by the expected cycle length:

$$ \text{long-run average rate} = \frac{E[\text{cost per cycle}]}{E[\text{cycle length}]}. $$

This holds even when both numerator and denominator are random, which is exactly the situation once replenishment timing becomes uncertain. In the classical model every cycle has the same deterministic length $Q/D$, so the theorem is invisible; the moment cycle length becomes random it becomes the central instrument.

**Modeling random supply availability.** The standard way to represent an intermittently available supplier is a two-state continuous-time Markov chain. The supplier sits in the "up" state for an exponentially distributed time with rate $\lambda$ (the *disruption rate*) before failing into the "down" state, where it stays for an exponentially distributed time with rate $\psi$ (the *recovery rate*) before returning. Mean up-time is $1/\lambda$, mean down-time is $1/\psi$, and the long-run fraction of time the supplier is down is $\lambda/(\lambda+\psi)$. The exponential assumption buys the memoryless property: a supplier observed to be down has the same remaining-downtime distribution no matter how long it has already been down, which keeps the analysis tractable. A useful consequence is the transient probability that the chain, started in the up state, is found in the down state $t$ time units later:

$$ \phi(t) = \frac{\lambda}{\lambda+\psi}\left(1 - e^{-(\lambda+\psi)t}\right). $$

At $t=0$ this is $0$ (the chain starts up) and as $t\to\infty$ it relaxes to the stationary down-probability $\lambda/(\lambda+\psi)$.

**The motivating phenomenon.** Two facts about existing systems frame the problem. First, under always-available supply, ZIO is provably optimal — there is never a reason to hold stock past the point where you can instantly refill. Second, once supply can fail, this stops being true: holding extra inventory becomes valuable precisely because it can carry the firm through a down period. So the design space opens up: if the penalty for unmet demand is high enough, order quantities should grow, and the question is by how much, and at what cost.

## Baselines

**Classical EOQ ($Q_E=\sqrt{2FD/h}$).** The thing to beat, and the thing that any disruption-aware order quantity must reduce to when the disruption rate goes to zero. Its gap: it assumes the supplier never fails, so it can under-order when supply is unreliable and unmet demand is costly, leaving the firm exposed during down periods. Used as the reference point — when the disruption-aware order quantity exceeds $Q_E$, the difference is the safety stock the firm holds to buffer against supply unavailability.

**Numerical minimization of a renewal-reward cost.** Once the cost rate is written as a function of $Q$ via renewal-reward, one can in principle just minimize it numerically (golden-section or bisection, given the function is single-troughed). This is the honest, exact route. Its limitation is not accuracy but form: the cost rate contains a transcendental term $e^{-(\lambda+\psi)Q/D}$, so the minimizer has no closed form. That matters when the order quantity must be embedded inside a larger model (multi-location systems, network design) where a clean algebraic expression is needed, or when one wants comparative-statics insight that a number from a solver does not provide.

## Evaluation settings

The natural test instances are small single-item systems parameterized by demand rate $D$, fixed order cost $F$, unit cost $a$, holding cost $h$, per-unit stockout penalty $\pi$, and the supplier's disruption and recovery rates $\lambda,\psi$. Representative ranges: $F$ of a few units, $a$ a few units, $\pi$ several times $a$, $h$ a small fraction, and disruption/recovery rates spanning frequent-short to rare-long down periods (e.g. mean up-time around a year, mean down-time around a month). The yardsticks are the long-run average cost per unit time $I(Q)$, the optimal order quantity, and — to read off the value of accounting for disruptions — the cost of using the classical $Q_E$ in a world that actually has disruptions, and (for any approximate order quantity) its relative error against the exact optimum in both $Q$ and cost.

## Code framework

The pieces that already exist: the classical EOQ formula, an exponential function, and a one-dimensional minimizer for a single-troughed (quasiconvex) function. The slots to be filled are the cost-rate function under random supply and any closed-form order quantity that may fall out of it.

```python
import math

def classical_eoq(D, F, h):
    return math.sqrt(2.0 * F * D / h)

def expected_cycle_length(Q, D, lam, psi):
    # TODO: cycle length is random once the supplier can be down at reorder;
    # derive E[T] from the supply model and the depletion time Q/D.
    pass

def cost_rate(Q, D, F, a, h, pi, lam, psi):
    # TODO: long-run average cost per unit time via renewal-reward,
    # = E[cost per cycle] / E[cycle length], as a function of Q.
    pass

def optimize_exact(D, F, a, h, pi, lam, psi):
    # golden-section / bisection over the single-troughed cost_rate
    pass

def approximate_order_quantity(D, F, a, h, pi, lam, psi):
    # TODO: a closed-form order quantity, if the cost rate can be
    # simplified to something with an algebraic minimizer.
    pass
```
