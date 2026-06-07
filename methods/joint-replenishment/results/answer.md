# The Joint Replenishment Problem: basic-period cyclic policies

## Problem

$M$ items share a supplier/line/shipment. Every replenishment incurs one **major** setup cost $A$ regardless of which items it carries; each item $i$ included also incurs a **minor** setup $a_i$. Demand is constant at rate $D_i$, holding cost is $h_i$ per unit per unit time, no shortages. Minimize the long-run average of (shared major + per-item minor + per-item holding) cost. Ordering each item independently by its own economic order quantity pays $A$ at every order of every item, forfeiting the economy of scope; the goal is to coordinate the order timings so one replenishment serves many items at once.

## Key idea

Restrict attention to **basic-period cyclic policies**: a basic cycle length $T$ with replenishment opportunities at $0,T,2T,\dots$ (the major cost $A$ paid once per opportunity), and each item $i$ ordered every $k_i$-th opportunity for a positive integer $k_i$ (cycle $k_iT$, batch $D_ik_iT$). Forcing every cycle to be an integer multiple of one common $T$ is what makes items' orders coincide so they share $A$. The cost is

$$
TC(T,\mathbf k)=\frac{A}{T}+\sum_{i=1}^{M}\!\left[\frac{a_i}{k_iT}+\frac{1}{2}h_ik_iD_iT\right]
=\frac{1}{T}\Big(A+\sum_i\frac{a_i}{k_i}\Big)+\frac{T}{2}\sum_i k_ih_iD_i .
$$

Two facts make this tractable. (1) **For fixed $\mathbf k$ the cost is economic-order-quantity-shaped in $T$** (a $1/T$ term plus a linear term), strictly convex, with closed-form minimizer

$$
T^*(\mathbf k)=\sqrt{\frac{2\big(A+\sum_i a_i/k_i\big)}{\sum_i k_ih_iD_i}} .
$$

(2) **For fixed $T$ the cost separates over items** (the major term is $\mathbf k$-independent), so each $k_i$ is chosen alone by minimizing the convex $f_i(k)=a_i/(kT)+\tfrac12 h_iD_iT\,k$. Its continuous optimum is $k_i^{\text{cont}}=\tfrac1T\sqrt{2a_i/(h_iD_i)}=T_i^{\text{EOQ}}/T$ — the ratio of item $i$'s own EOQ cycle to the basic period — rounded to the integer $\ge1$ that minimizes $f_i$.

Alternating (1) and (2) is a monotone coordinate descent converging to a local optimum (the RAND/iterative heuristic); sweeping the finitely many $T$-values where some $k_i$ flips (the breakpoints $T=T_i^{\text{EOQ}}/\sqrt{k(k+1)}$) yields the global optimum of the family in near-linear time. Restricting the multipliers to **powers of two**, $k_i\in\{1,2,4,\dots\}$, makes all intervals nested (every pair of items perfectly aligned) and, with the base period $T$ optimized continuously, is provably within $\approx2\%$ of the true continuous-time optimum (within $\approx6\%$ if $T$ is fixed) — the worst-case loss per item being $\tfrac12(\sqrt2+1/\sqrt2)\approx1.0607$ from the flatness of the EOQ cost. A single item, or $A$-only, recovers the economic order quantity $T^*=\sqrt{2K/(hD)}$.

## Algorithm

1. Initialize $k_i=1$ for all $i$; set $T\leftarrow T^*(\mathbf k)$.
2. **k-step:** for each item set $k_i$ to the integer (or, in the power-of-two variant, the power of two) minimizing $f_i(k)=a_i/(kT)+\tfrac12 h_iD_iT\,k$, i.e. round $T_i^{\text{EOQ}}/T$.
3. **T-step:** set $T\leftarrow T^*(\mathbf k)$ by the closed form.
4. Repeat 2–3 until $\mathbf k$ is unchanged. Return per-item intervals $k_iT$ and total cost $TC(T,\mathbf k)$.
5. (Global option) enumerate the $T$-breakpoints, evaluate the closed-form $T^*$ on each interval, keep the cheapest.

## Code

```python
import math, itertools

def jrp_cost(T, k, A, a, h, D):
    # major shared once per basic cycle; minor every k_i*T; holding on avg inventory D_i*k_i*T/2
    major = A / T
    minor = sum(a[i] / (k[i] * T) for i in range(len(k)))
    hold  = (T / 2.0) * sum(k[i] * h[i] * D[i] for i in range(len(k)))
    return major + minor + hold

def optimal_T(k, A, a, h, D):
    # fixed k: T* = sqrt(2(A + sum a_i/k_i) / sum k_i h_i D_i)
    num = 2.0 * (A + sum(a[i] / k[i] for i in range(len(k))))
    den = sum(k[i] * h[i] * D[i] for i in range(len(k)))
    return math.sqrt(num / den)

def best_k_given_T(T, a, h, D):
    # fixed T: cost separates; minimize each convex f_i(k)=a_i/(kT)+0.5 h_i D_i T k over k>=1
    k = []
    for i in range(len(a)):
        kc = math.sqrt(2.0 * a[i] / (h[i] * D[i] * T * T)) if a[i] > 0 else 0.0  # = T_i^EOQ / T
        kk = max(1, int(math.floor(kc)))
        f = lambda kv: a[i] / (kv * T) + 0.5 * h[i] * D[i] * kv * T
        while f(kk + 1) < f(kk):
            kk += 1
        while kk > 1 and f(kk - 1) < f(kk):
            kk -= 1
        k.append(kk)
    return k

def iterative_jrp(A, a, h, D, max_iter=100):
    k = [1] * len(a)
    T = optimal_T(k, A, a, h, D)
    for _ in range(max_iter):
        kn = best_k_given_T(T, a, h, D)
        if kn == k:
            break
        k = kn
        T = optimal_T(k, A, a, h, D)
    T = optimal_T(k, A, a, h, D)
    return T, k, [k[i] * T for i in range(len(k))], jrp_cost(T, k, A, a, h, D)

def power_of_two_jrp(A, a, h, D, max_iter=100):
    def to_pow2(x):
        if x < 1: return 1
        lo = 2 ** int(math.floor(math.log2(x)))
        return lo if (x / lo) <= ((2 * lo) / x) else 2 * lo   # nearest power of two in log scale
    k = [1] * len(a)
    T = optimal_T(k, A, a, h, D)
    for _ in range(max_iter):
        kn = [to_pow2(math.sqrt(2.0 * a[i] / (h[i] * D[i] * T * T)) if a[i] > 0 else 1.0)
              for i in range(len(a))]
        if kn == k:
            break
        k = kn
        T = optimal_T(k, A, a, h, D)
    T = optimal_T(k, A, a, h, D)
    return T, k, [k[i] * T for i in range(len(k))], jrp_cost(T, k, A, a, h, D)

def brute_force(A, a, h, D, kmax=12):
    # bounded reference: enumerate integer multipliers in [1,kmax]^M, closed-form T for each
    best = None
    for k in itertools.product(range(1, kmax + 1), repeat=len(a)):
        T = optimal_T(list(k), A, a, h, D)
        c = jrp_cost(T, list(k), A, a, h, D)
        if best is None or c < best[0]:
            best = (c, T, list(k))
    return best

def coordinate_items(A, a, h, D):
    # entry point: coordinate M items sharing major cost A via the basic-period policy
    # -> per-item order intervals k_i*T and the total average cost
    T, k, intervals, cost = iterative_jrp(A, a, h, D)
    return intervals, cost

if __name__ == "__main__":
    A = 100.0
    a = [10.0, 20.0, 5.0]
    h = [1.0, 0.5, 2.0]
    D = [100.0, 50.0, 200.0]
    Ti, ki, intervals, ci = iterative_jrp(A, a, h, D)
    Tp, kp, _, cp        = power_of_two_jrp(A, a, h, D)
    cb, Tb, kb           = brute_force(A, a, h, D)
    print("iterative:", round(Ti, 4), ki, "cost", round(ci, 4), "intervals", [round(x,4) for x in intervals])
    print("pow2     :", round(Tp, 4), kp, "cost", round(cp, 4))
    print("brute    :", round(Tb, 4), kb, "cost", round(cb, 4))
    # single item with no minor cost reduces to EOQ
    assert abs(optimal_T([1], 50.0, [0.0], [1.0], [100.0]) - math.sqrt(2*50/(1*100))) < 1e-12
```

On the example the iterative heuristic returns $\mathbf k=(1,2,1)$, $T\approx0.6742$, cost $\approx370.81$; the power-of-two variant returns the same $\mathbf k$ and cost; brute-force enumeration confirms this is the global optimum, and the single-item assertion recovers the economic order quantity.
