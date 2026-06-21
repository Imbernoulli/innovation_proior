# Context: coordinating the replenishment of many items that share one setup cost

## Research question

A warehouse (or a production line, or an inbound shipping lane) stocks $M$ distinct items, each with a steady, known demand rate $D_i$ and a holding cost $h_i$ per unit per unit time. Items are not independent at the moment of ordering: every replenishment, no matter which items it carries, triggers one shared **major** setup cost $A$ — a truck dispatched, a container booked, a line changed over, a purchase order opened with the supplier. On top of that, each individual item $i$ included in a replenishment carries a smaller **minor** setup cost $a_i$ — its own order line, its own receiving and put-away. The question is how to schedule replenishments of all $M$ items together so that total long-run average cost — the shared major cost, the per-item minor costs, and the per-item holding costs — is as small as possible.

What makes this more than $M$ separate problems is precisely the shared $A$. Two items whose natural order times fall close together could ride on the same truck and split one $A$. So the heart of the question is *coordination* — choosing the order timings to let many items share each payment of $A$ — traded off against the fact that forcing an item onto a common rhythm can pull its order quantity away from what it would individually prefer, inflating its holding or minor-setup cost.

## Background

**The single-item root: the economic order quantity.** The atom of the whole subject is one item in isolation, demand rate $D$, a fixed cost $K$ charged per order, holding cost $h$ per unit per unit time, no shortages. Order in batches of size $Q$; a batch lasts $Q/D$ time, so orders arrive at rate $D/Q$ and average on-hand inventory is $Q/2$. Average cost per unit time is

$$C(Q)=\frac{D}{Q}\,K+\frac{Q}{2}\,h,$$

a convex function: ordering cost falls like $1/Q$, holding cost rises linearly. Setting $C'(Q)=0$ gives the economic order quantity $Q^*=\sqrt{2KD/h}$ (Harris 1913, "How Many Parts to Make at Once"). It is cleaner still in the time domain. Writing the cycle time $T=Q/D$, average inventory is $\tfrac12 D T$, orders come at rate $1/T$, and

$$C(T)=\frac{K}{T}+\frac{1}{2}hD\,T,\qquad T^*=\sqrt{\frac{2K}{hD}}.$$

This square-root law — balance a $1/T$ ordering term against a linear $T$ holding term — is the lever everything below pulls on. It says each item, left alone, has a *natural cycle* $T_i=\sqrt{2a_i/(h_iD_i)}$ set by its own minor cost, demand, and holding rate.

**Independent ordering.** Run EOQ separately for each item: item $i$ orders every $T_i=\sqrt{2(A+a_i)/(h_iD_i)}$, paying $A$ at each of its own order epochs.

**Common-cycle policy.** The simplest coordination: force *all* items onto one shared cycle $T$, every item ordered every time, so the major cost is shared by everyone at every order. Cost $A/T+\sum_i[a_i/T+\tfrac12h_iD_iT]$, optimized at $T=\sqrt{2(A+\sum_ia_i)/\sum_ih_iD_i}$.

**An event-driven alternative on the table.** Instead of a fixed schedule, coordination can be triggered reactively. Balintfy (1964), "On a basic class of multi-item inventory problems," proposed a continuous-review **can-order policy**: each item carries three levels, a must-order point $s_i$, a can-order point $c_i\ (s_i<c_i<S_i)$, and an order-up-to level $S_i$. Whenever any item's inventory falls to its must-order point $s_i$, a replenishment is triggered (paying $A$); at that same moment every other item currently at or below its can-order point $c_i$ is dragged into the order and topped up to $S_i$, so it rides along on the shared $A$. Coordination here is opportunistic — items sharing a setup because they happen to be low together — rather than scheduled.

## Baselines

**Per-item EOQ (Harris 1913), applied independently.** Each item solves its own square-root problem with cost $K_i=A+a_i$, giving $T_i=\sqrt{2(A+a_i)/(h_iD_i)}$ and $Q_i=D_iT_i$.

**Common-cycle policy.** All items ordered on one shared cycle $T$; major cost shared by all at every order. Optimal $T=\sqrt{2(A+\sum_ia_i)/\sum_ih_iD_i}$.

**Can-order $(s_i,c_i,S_i)$ policy (Balintfy 1964).** Continuous-review, event-driven coordination: must-order triggers a setup, can-order items ride along.

## Evaluation settings

The natural test instance is a deterministic, constant-demand, infinite-horizon multi-item system specified by: the number of items $M$; the shared major setup cost $A$; a minor setup cost $a_i$, a holding cost $h_i$, and a demand rate $D_i$ for each item $i$. A candidate policy assigns each item a replenishment schedule; the yardstick is the long-run average total cost it induces — shared major, per-item minor, per-item holding. For a small instance the policy's cost should be checkable against an exact reference optimum so that any fast heuristic can be validated. The relevant outputs are the per-item order intervals and the total average cost; a single item, or $A=0$, must reproduce the economic-order-quantity solution.

## Code framework

The pre-existing primitives are entirely from the single-item world: an economic-order-quantity-style cost evaluator that, for one item on a cycle, trades a fixed cost against linear holding, and its square-root minimizer. What does not yet exist is any mechanism for coupling the items through the shared major cost.

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
    # TODO: couple the items through the shared major cost A and return each item's
    #       order interval together with the total long-run average cost.
    pass
```
