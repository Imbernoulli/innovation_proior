# Context: coordinating the replenishment of many items that share one setup cost

## Research question

A warehouse (or a production line, or an inbound shipping lane) stocks $M$ distinct items, each with a steady, known demand rate $D_i$ and a holding cost $h_i$ per unit per unit time. Items are not independent at the moment of ordering: every replenishment, no matter which items it carries, triggers one shared **major** setup cost $A$ — a truck dispatched, a container booked, a line changed over, a purchase order opened with the supplier. On top of that, each individual item $i$ included in a replenishment carries a smaller **minor** setup cost $a_i$ — its own order line, its own receiving and put-away. The question is how to schedule replenishments of all $M$ items together so that total long-run average cost — the shared major cost, the per-item minor costs, and the per-item holding costs — is as small as possible.

What makes this more than $M$ separate problems is precisely the shared $A$. If the items were ordered on independent schedules, the major cost $A$ would be paid at every one of their separate order epochs, which is wasteful: two items whose natural order times fall close together could have ridden on the same truck and split one $A$. So the heart of the question is *coordination* — choosing the order timings to let many items share each payment of $A$ — traded off against the fact that forcing an item onto a common rhythm can pull its order quantity away from what it would individually prefer, inflating its holding or minor-setup cost. A planner wants a rule that is cheap to compute, easy to audit, and provably close to the true optimum, not an opaque schedule pulled from a black box.

## Background

**The single-item root: the economic order quantity.** The atom of the whole subject is one item in isolation, demand rate $D$, a fixed cost $K$ charged per order, holding cost $h$ per unit per unit time, no shortages. Order in batches of size $Q$; a batch lasts $Q/D$ time, so orders arrive at rate $D/Q$ and average on-hand inventory is $Q/2$. Average cost per unit time is

$$C(Q)=\frac{D}{Q}\,K+\frac{Q}{2}\,h,$$

a convex function: ordering cost falls like $1/Q$, holding cost rises linearly. Setting $C'(Q)=0$ gives the economic order quantity $Q^*=\sqrt{2KD/h}$ (Harris 1913, "How Many Parts to Make at Once"). It is cleaner still in the time domain. Writing the cycle time $T=Q/D$, average inventory is $\tfrac12 D T$, orders come at rate $1/T$, and

$$C(T)=\frac{K}{T}+\frac{1}{2}hD\,T,\qquad T^*=\sqrt{\frac{2K}{hD}}.$$

This square-root law — balance a $1/T$ ordering term against a linear $T$ holding term — is the lever everything below pulls on. It says each item, left alone, has a *natural cycle* $T_i=\sqrt{2a_i/(h_iD_i)}$ set by its own minor cost, demand, and holding rate.

**Why naively applying EOQ per item leaves money on the table.** Run EOQ separately for each item and item $i$ orders every $T_i=\sqrt{2(A+a_i)/(h_iD_i)}$ — but now $A$ is paid at *every* item's every order, $M$ independent streams of major-cost payments. Nothing is shared. Since the natural cycles $T_i$ differ across items, their order epochs almost never coincide, so almost every replenishment carries just one item and pays a full $A$ for it. The shared structure of the cost — one $A$ could cover many items at once — is completely unused. The pre-method fact this rests on: the major cost is a genuine economy of scope, and independent ordering forfeits it.

**The coordination idea and its tension.** To share $A$, order items on a *common rhythm*. Pick a basic cycle length $T$ and let every replenishment opportunity fall at times $0,T,2T,3T,\dots$; pay the major cost $A$ once per basic cycle. Then have each item $i$ replenish only on a subset of these opportunities — every $k_i$-th one, for some positive integer $k_i$ — so item $i$'s cycle is $k_iT$. Now any items that order on the same opportunity automatically share that opportunity's $A$. The tension is immediate and is the crux of the design: a small $T$ pays $A$ often (good for items that want frequent orders, bad for the shared cost); forcing item $i$ onto $k_iT$ when $k_iT$ differs from its natural $T_i$ raises its own minor-plus-holding cost. The optimum is a compromise between amortizing $A$ across items and respecting each item's own EOQ rhythm.

**The combinatorial difficulty.** Once the policy is parameterized by a basic period $T$ and a vector of integer multiples $(k_1,\dots,k_M)$, the cost is smooth in $T$ but the $k_i$ are integers — a mixed continuous/integer optimization. For fixed integers the $T$-problem is a one-dimensional convex problem with a square-root solution, but the integer vector lives in a combinatorially large space, and the two are coupled (the best $T$ depends on the $k_i$ and the best $k_i$ depends on $T$). Sweeping all integer vectors is exponential in $M$. So the open problem is to find a restriction of the integer choices, or an iteration over them, that is both computationally cheap and provably near the true continuous-time optimum.

**An event-driven alternative on the table.** Instead of a fixed schedule, coordination can be triggered reactively. Balintfy (1964), "On a basic class of multi-item inventory problems," proposed a continuous-review **can-order policy**: each item carries three levels, a must-order point $s_i$, a can-order point $c_i\ (s_i<c_i<S_i)$, and an order-up-to level $S_i$. Whenever any item's inventory falls to its must-order point $s_i$, a replenishment is triggered (paying $A$); at that same moment every other item currently at or below its can-order point $c_i$ is dragged into the order and topped up to $S_i$, so it rides along on the shared $A$. Coordination here is opportunistic — items sharing a setup because they happen to be low together — rather than scheduled. It addresses stochastic demand naturally, but it leaves a hard joint optimization over $3M$ thresholds whose interactions are awkward to characterize.

## Baselines

**Per-item EOQ (Harris 1913), applied independently.** Each item solves its own square-root problem with cost $K_i=A+a_i$, giving $T_i=\sqrt{2(A+a_i)/(h_iD_i)}$ and $Q_i=D_iT_i$. Core idea and exact target the coordinated theory must beat and must reduce to (one item, or $A=0$, recovers it). Gap: it pays the shared major cost $A$ on every item's every order, ignoring that one replenishment can cover many items; the cross-item economy of scope is entirely forfeited, so it is systematically too expensive whenever $A$ is non-trivial.

**Common-cycle policy ($k_i\equiv1$).** The simplest coordination: force *all* items onto one shared cycle $T$, every item ordered at every opportunity, so the major cost is shared by everyone every time. Cost $A/T+\sum_i[a_i/T+\tfrac12h_ik_iD_iT]$ with all $k_i=1$, optimized at $T=\sqrt{2(A+\sum_ia_i)/\sum_ih_iD_i}$. Core idea: maximal sharing of $A$. Gap: it over-orders the slow, cheap-to-hold items (a bulky low-demand item is forced to the same frequency as a fast one), inflating their holding cost; it is only optimal when the items are similar, and can be far from optimal when demand rates or holding costs are heterogeneous — which is exactly when coordination matters most.

**Can-order $(s_i,c_i,S_i)$ policy (Balintfy 1964).** Continuous-review, event-driven coordination: must-order triggers a setup, can-order items ride along. Core idea above. Gap: the joint optimization over three thresholds per item is analytically intractable in general and relies on approximations/heuristics; for deterministic constant demand it is heavier machinery than a fixed cyclic schedule and offers no clean optimality guarantee.

## Evaluation settings

The natural test instance is a deterministic, constant-demand, infinite-horizon multi-item system specified by: the number of items $M$; the shared major setup cost $A$; a minor setup cost $a_i$, a holding cost $h_i$, and a demand rate $D_i$ for each item $i$. A candidate policy is a basic cycle length $T>0$ together with a vector of positive integers $(k_1,\dots,k_M)$, meaning item $i$ is replenished every $k_iT$; its order quantity is then $D_ik_iT$. The yardstick is the long-run average total cost — shared major, per-item minor, per-item holding — evaluated on this policy, and, for a small instance, the exact minimum obtained by enumerating the integer multipliers over a bounded range (computing the closed-form best $T$ for each integer vector) so that any fast heuristic can be checked against the true optimum. The relevant outputs are the per-item order intervals $k_iT$ and the total average cost; a single item, or $A=0$, must reproduce the economic-order-quantity solution.

## Code framework

The pre-existing primitives are entirely from the single-item world: an economic-order-quantity-style cost evaluator that, for one item on a cycle, trades a fixed cost against linear holding, and its square-root minimizer. What does not yet exist is any mechanism for coupling the items through the shared major cost — the rule that decides how the basic period and the per-item frequencies should be set together.

```python
import math

def eoq_cost_one_item(T, fixed, h, D):
    """Single-item average cost on cycle T: fixed/T + (1/2) h D T (EOQ shape)."""
    return fixed / T + 0.5 * h * D * T

def eoq_cycle(fixed, h, D):
    """Square-root minimizer of the single-item EOQ cost: T* = sqrt(2 fixed/(h D))."""
    return math.sqrt(2.0 * fixed / (h * D))

def coordinate_items(A, a, h, D):
    """Choose a shared replenishment schedule for M items that share the major cost A.

    Inputs: shared major cost A; per-item minor cost a[i], holding h[i], demand D[i].
    Should return per-item order intervals and the total average cost.
    """
    # TODO: define the average cost of a coordinated (basic-period + integer-multiple)
    #       policy, and decide how the basic period and the per-item frequencies are
    #       set together so the shared major cost is amortized across items.
    pass
```
