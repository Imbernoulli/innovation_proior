# JRP synthesis

## Pain point
Multiple items (SKUs) drawn from a shared supplier / shared production line / shared shipment.
Ordering ANY item incurs a shared MAJOR setup cost A (truck dispatch, line changeover, container).
Each item additionally incurs a MINOR setup cost a_i (order line, per-SKU handling) and a holding cost.
If you EOQ each item independently you pay the major cost A every time any item orders — wasteful, because
two items that order at nearby times could have shared one truck. The coordination motive: synchronize orders
so the major cost is amortized across items.

## Single-item ancestor: EOQ (Harris 1913, "How Many Parts to Make at Once", Factory mag)
One item, constant demand rate D, fixed order cost K, holding cost h per unit per year.
Order quantity Q, cycle T=Q/D. Average cost per unit time:
  C(Q) = (D/Q)K + (Q/2)h
  -> Q* = sqrt(2KD/h),  T* = Q*/D = sqrt(2K/(hD)).
Balances ordering cost (decreasing in Q) vs holding cost (increasing in Q). Convex in Q.
Equivalently in cycle time: C(T) = K/T + (1/2)hD T, T* = sqrt(2K/(hD)). This is the atom.
Verified against multiple sources (Harris 1913; standard).

## The JRP standard formulation (from Porras & Dekker EUR EI2003-52, p.2; canonical "(k_j,T) policy")
M items. Major cost A per replenishment opportunity. Minor cost a_j per item ordered.
Basic cycle T (a replenishment opportunity every T). Item j ordered every k_j*T (k_j positive integer).
So item j orders with frequency 1/(k_j T), paying a_j each time. Major A paid once per basic cycle (frequency 1/T)
under the STRICT/basic-period policy (charged every basic period even if a cycle is "empty"; the
correction-factor variant only charges A when at least one item actually orders).

Average total cost (strict basic-period model):
  TC(T,k) = A/T + Σ_j [ a_j/(k_j T) + (1/2) h_j k_j D_j T ]
          = ( A + Σ_j a_j/k_j ) / T  +  (T/2) Σ_j k_j h_j D_j.

Holding: item j ordered every k_j T, demand D_j, so order size = D_j k_j T, average inventory = (1/2) D_j k_j T,
holding rate = (1/2) h_j D_j k_j T. Correct.
Ordering: major A/T + minor a_j/(k_j T). Correct.

This matches EUR p.2 "JRP (standard formulation)": TC = A/T + Σ_j [ a_j/(k_j T) + (1/2) h_j k_j D_j T ].
The MOQ specialization (their main model) drops the a_j term.

## Closed-form T* given k  (EUR eq.4, generalized to minor costs)
For fixed k, TC is strictly convex in T (sum of c1/T + c2*T with c1,c2>0). dTC/dT=0:
  -(A + Σ a_j/k_j)/T^2 + (1/2)Σ k_j h_j D_j = 0
  => T* = sqrt( 2(A + Σ_j a_j/k_j) / Σ_j k_j h_j D_j ).
EUR eq.(4) is the no-minor-cost case: T* = sqrt(2A / Σ h_j k_j D_j). Same with a_j=0. VERIFIED.

## k_j update given T (the iterative heuristic core)
For fixed T, TC separates over j (major A/T independent of k). Minimize per item:
  f_j(k_j) = a_j/(k_j T) + (1/2) h_j k_j D_j T  over positive integer k_j.
Continuous relaxation: f_j'(k)=0 -> a_j/(k^2 T) = (1/2) h_j D_j T
  => k_j^cont = sqrt( 2 a_j / (h_j D_j T^2) ) = (1/T) sqrt( 2 a_j/(h_j D_j) ).
Note sqrt(2a_j/(h_j D_j)) is exactly item j's standalone EOQ cycle time, call it T_j^EOQ. So
  k_j^cont = T_j^EOQ / T.  Beautiful: the multiplier is the ratio of item j's natural cycle to the basic cycle.
Round to nearest integer >= 1 by the convexity rule: choose integer k minimizing f_j, i.e. the
integer k with f_j(k) <= f_j(k+1) and f_j(k) <= f_j(k-1). Equivalently pick smallest k>=1 with
  f_j(k+1) >= f_j(k)  <=>  k(k+1) >= 2a_j/(h_j D_j T^2) = (T_j^EOQ/T)^2.
(Standard rounding for convex integer programs: round at the geometric/threshold boundary.)

## Iterative algorithm (Goyal 1974 enumeration; Silver 1976; Viswanathan; EUR §3.1)
Step 0: k_j=1 all j.
Step 1: given k, compute T* by closed form.
Step 2: given T*, update each k_j to its optimal integer (rounding rule above).
Repeat 1-2 until k stops changing. Converges fast; gives a local optimum.
To make GLOBAL: T* is bounded; partition T-axis into intervals where k(T) constant (TC piecewise convex
in T), enumerate. Goyal's method enumerates over the smallest multiplier / over T-ranges.

## Power-of-two policies (Roundy 1985 "98%-effective integer-ratio lot-sizing for one-warehouse multi-retailer";
## Federgruen-Queyranne-Zheng 1992 generalize to submodular joint setup; root: Brown 1978 for single-item EOQ)
Restrict k_j to powers of two: k_j in {1,2,4,8,...}. Then ANY two replenishment intervals are nested
(one divides the other) -> truck-sharing is automatic / schedules align perfectly, and the combinatorial
search collapses. Worst-case: best power-of-two policy within 2% of optimum when base period T is
optimized continuously (98%-effective); within 6% (94%) if T is fixed. The 6% bound traces to Brown (1978)
for single-item EOQ: rounding the EOQ cycle to the nearest power-of-two of a fixed base loses at most ~6%.
Why 2% for variable base: optimizing T adds a degree of freedom that recovers most of the loss.
The power-of-two restriction is what makes the basic-period approach both tractable and provably near-optimal.

Brown/Roundy single-item bound intuition: cost ratio C(2^m T_0)/C(T*) for the EOQ-shape cost
g(t)=c1/t + c2 t is g(rT*)/g(T*) = (1/2)(r + 1/r) where r is the rounding ratio in (1/sqrt2, sqrt2].
Max of (1/2)(r+1/r) over r in [1/sqrt2, sqrt2] is at endpoints: (1/2)(sqrt2 + 1/sqrt2)=(1/2)(3/sqrt2)=3/(2sqrt2)
= 1.0606..., i.e. 6.066% worst case. VERIFIED arithmetic. With T optimized over the joint cost the bound improves to 2%.

## Can-order (s,c,S) policy (Balintfy 1964, "On a basic class of multi-item inventory problems", Mgmt Sci)
Alternative to cyclic: continuous review, stochastic. Each item has must-order point s, can-order point c, order-up-to S.
When any item hits s it triggers an order; every other item currently below its can-order point c is "dragged along"
and topped up to S, so they share the major setup. Event-driven coordination instead of schedule-driven.
Gap: hard to optimize jointly (must-order/can-order/order-up-to per item), heuristic-laden; deterministic cyclic
basic-period policy is cleaner and provably near-optimal for constant demand.

## Complexity note (knowable framing): JRP is hard combinatorially (the integer k search). Roundy's power-of-two
gives the practical resolution. (The strong-NP-hardness proof is posterior — do NOT cite as derivation-time.)

## In-frame discipline
Target = the JRP basic-period / k_i-multiplier method + power-of-two restriction + RAND/iterative heuristic.
EOQ (Harris 1913) is the ancestor and MAY be cited. Do not cite Goyal/Shu/Roundy/Balintfy AS artifacts in
reasoning.md (they ARE the method being discovered). In answer.md may name "joint replenishment problem",
"basic-period policy", "power-of-two", "RAND/iterative heuristic", "can-order policy" as the things built.

## Code plan
- jrp_cost(T, k, A, a, h, D) -> total avg cost  (the formula)
- optimal_T(k, A, a, h, D) -> closed form sqrt(2(A+Σa/k)/Σ k h D)
- best_k_given_T(T, a, h, D) -> per-item integer minimizer (rounding rule)
- iterative_jrp(...) -> alternate T<->k to convergence, return T, k, intervals k*T, cost
- power_of_two_jrp(...) -> restrict k to powers of two (round k_cont to nearest power of 2 by cost), optimize base T
- validate: brute-force enumerate k in small ranges on a tiny instance, compare to iterative & pow2.
Scaffold (context): generic per-item EOQ-style cost evaluator + an empty "coordinate the items" slot.
