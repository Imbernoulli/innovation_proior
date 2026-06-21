We have $M$ items that share a single supply channel — one truck, one container, one production line, one purchase order — and the awkward fact that every replenishment, whatever it carries, costs a shared **major** setup $A$, while each item $i$ riding on that order adds its own smaller **minor** setup $a_i$. Demand is steady and known at rate $D_i$, holding costs $h_i$ per unit per unit time, no shortages, and I want the schedule that minimizes the long-run average of major plus minor plus holding cost. The atom I start from is the single-item economic order quantity: one item with fixed cost $K$ per order has average cost $\frac{D}{Q}K+\frac{Q}{2}h$, convex, minimized at $Q^*=\sqrt{2KD/h}$, or in the time domain — cycle $T=Q/D$, average inventory $\tfrac12 DT$ — at $T^*=\sqrt{2K/(hD)}$, the square-root law that balances a $1/T$ ordering term against a linear holding term. The obvious move is to run this per item with $K_i=A+a_i$, giving $T_i=\sqrt{2(A+a_i)/(h_iD_i)}$. But that throws money away: each item then pays a full $A$ at every one of its orders, and because the natural cycles $T_i$ are generically incommensurate, the order epochs essentially never coincide, so almost every dispatched truck carries one item and eats a whole $A$ for it. The major cost is an economy of scope — one order could cover items 1, 2 and 5 and split a single $A$ — and independent EOQ forfeits all of it. The opposite extreme, forcing every item onto one common cycle so all share $A$ at every order, over-orders the slow, expensive-to-hold items and is only good when items are similar — which is exactly when coordination matters least. An event-driven can-order $(s_i,c_i,S_i)$ policy coordinates opportunistically but leaves an intractable joint optimization over three thresholds per item, heavier than needed when demand is deterministic and order times are predictable anyway.

I propose a **basic-period cyclic policy**. Fix one basic cycle length $T$ so replenishment opportunities fall on the grid $0,T,2T,\dots$, with the major cost $A$ paid once per opportunity, and let each item $i$ order on every $k_i$-th opportunity for a positive integer $k_i$ — cycle $k_iT$, batch $D_ik_iT$, average inventory $\tfrac12 D_ik_iT$. The integer constraint is the load-bearing design choice: if $k_i$ were any real number the items would drift back out of phase and stop sharing, but pinning every cycle to an integer multiple of one common $T$ forces all order epochs onto a single shared grid, so whenever two items' order times coincide they coincide exactly on a grid point and split that opportunity's $A$, never landing a hair apart and missing each other. The cost of this policy is

$$ TC(T,\mathbf k)=\frac{A}{T}+\sum_{i=1}^{M}\!\left[\frac{a_i}{k_iT}+\frac{1}{2}h_ik_iD_iT\right]=\frac{1}{T}\Big(A+\sum_i\frac{a_i}{k_i}\Big)+\frac{T}{2}\sum_i k_ih_iD_i, $$

and both naive endpoints live inside it: one item with $k_1=1$ gives back the EOQ with fixed cost $A+a_1$, and all $k_i=1$ is the common-cycle policy — the $k_i>1$ freedom is precisely what lets slow items skip and breathe while fast ones keep the base rhythm.

What makes this tractable is that the two variables decouple cleanly in opposite directions. For **fixed $\mathbf k$** the cost is EOQ-shaped in $T$ — a $1/T$ term with numerator $\mathcal A(\mathbf k)=A+\sum_i a_i/k_i$ plus a linear term with coefficient $\mathcal H(\mathbf k)=\sum_i k_ih_iD_i$, strictly convex — so differentiating $-\mathcal A/T^2+\tfrac12\mathcal H=0$ gives the closed-form best basic period in one square root,

$$ T^*(\mathbf k)=\sqrt{\frac{2\big(A+\sum_i a_i/k_i\big)}{\sum_i k_ih_iD_i}}. $$

For **fixed $T$** the major term $A/T$ is constant in $\mathbf k$ and the rest is a sum over items with no cross terms, so the $M$-dimensional integer problem separates completely into $M$ independent one-dimensional ones: minimize each convex $f_i(k)=\frac{a_i}{kT}+\tfrac12 h_iD_iT\,k$ alone. Relaxing $k$ to reals, $f_i'(k)=0$ gives $k_i^{\text{cont}}=\frac1T\sqrt{2a_i/(h_iD_i)}=T_i^{\text{EOQ}}/T$ — the ideal multiplier is just the ratio of the item's own EOQ cycle to the basic period, telling each item how many basic cycles to wait so that $k_iT$ sits near its natural rhythm. Since $f_i$ is convex, the integer optimum is one of the two integers bracketing $k_i^{\text{cont}}$ (floored at 1), found by stepping from $\lfloor k_i^{\text{cont}}\rfloor$ toward whichever neighbor lowers $f_i$; the equivalent closed-form test is to pick the smallest $k\ge1$ with $k(k+1)\ge(T_i^{\text{EOQ}}/T)^2$. These two cheap maps are each other's complements, so I alternate them: start at $k_i=1$, compute $T^*(\mathbf 1)$, re-round each $k_i$ at that $T$, recompute $T^*$, and repeat until $\mathbf k$ stops changing. Each half-step exactly minimizes one block with the other held, so the cost decreases monotonically and is bounded below, and the iteration converges in a few steps — a coordinate descent (the RAND/iterative heuristic) that finds a local optimum.

Because coordinate descent can be trapped by its start, a global sweep is also available and cheap thanks to the structure of $T$. As $T$ shrinks each $k_i^{\text{cont}}=T_i^{\text{EOQ}}/T$ grows, so the optimal integer $k_i(T)$ is a non-decreasing step function; between its jumps $\mathbf k$ is constant and $TC$ is just the convex EOQ curve, making the lower envelope $\min_{\mathbf k}TC(T,\mathbf k)$ piecewise convex with breakpoints exactly where some $k_i$ flips, $T=T_i^{\text{EOQ}}/\sqrt{k(k+1)}$. On each interval $\mathbf k$ is known, so I drop in the closed-form $T^*(\mathbf k)$ clipped to the interval, read off the cost, and take the best across intervals — a global optimum of the family in time linear in the number of breakpoints, bounded above by the common-cycle period $\sqrt{2(A+\sum_i a_i)/\sum_i h_iD_i}$ (past which every item already wants $k_i=1$) and walked downward until $T^*(\mathbf k)$ stops landing inside its own interval. One modeling note: I charge $A$ once per basic cycle regardless of how many items show up, which slightly over-charges opportunities where few items order, but this conservative strict accounting is exactly what keeps the cost separable and upper-bounds the true cost; charging $A$ only on opportunities where some item orders would couple the items through their coincidence pattern and destroy the separability.

What makes the whole restriction justified — rather than merely convenient — is the flatness of the EOQ cost. For one item, being forced to use $t=r\,t^*$ instead of its optimum costs a ratio $\frac{g(rt^*)}{g(t^*)}=\frac12(r+1/r)$, which is quadratically small near $r=1$: a $\pm10\%$ error in the cycle costs under half a percent. This is the gift that makes coordination cheap — dragging an item slightly off its natural cycle barely hurts it. It also motivates a further sharpening: restrict the multipliers to **powers of two**, $k_i\in\{1,2,4,8,\dots\}$. Powers of two are nested, so any two intervals $2^aT$ and $2^bT$ have one dividing the other and every pair of items is perfectly aligned — whenever the slower orders, the faster orders too — collapsing the awkward "coincide only every 6 cycles" case and bringing the strict and true charging of $A$ much closer. Each $k_i$ is then just the nearest power of two to $k_i^{\text{cont}}$, at most a factor $\sqrt2$ off; plugging the worst case $r=\sqrt2$ into the flatness formula gives $\frac12(\sqrt2+\tfrac1{\sqrt2})=\frac{3}{2\sqrt2}\approx1.0607$, so with $T$ fixed no item — hence not the sum — loses more than about $6\%$. And since I am free to slide the one shared $T$ continuously to center the dyadic grid favorably across all items at once, that worst case drops to roughly $2\%$ of the true continuous-time optimum, not merely the best of the restricted family. So the exponential integer search is replaced by "round each item's ideal multiple to the nearest power of two and tune one $T$," provably near-optimal, with a single item or $A$-only recovering the economic order quantity $T^*=\sqrt{2K/(hD)}$ exactly.

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
