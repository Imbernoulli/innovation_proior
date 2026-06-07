Let me start from the pain, before reaching for any policy. I have $M$ items sharing a warehouse, each with a steady demand $D_i$, a holding cost $h_i$, and its own little order-line cost $a_i$. And there's this one annoying shared thing: every replenishment, no matter what's on it, costs me a major setup $A$ — a truck, a container, a changeover, a PO. I want to order all of these as cheaply as I can over the long run.

The instinct is to just solve each item by itself. That I know how to do — it's the economic order quantity. One item, fixed cost $K$ per order, demand $D$, holding $h$. If I order batches of $Q$, a batch covers $Q/D$ time, so I order at rate $D/Q$ and on average hold $Q/2$ units. Cost rate is $\frac{D}{Q}K + \frac{Q}{2}h$, convex, and the derivative $-\frac{DK}{Q^2}+\frac{h}{2}=0$ gives $Q^*=\sqrt{2KD/h}$. I like it better in time, though — write $T=Q/D$ for the cycle, average inventory $\tfrac12 DT$, ordering rate $1/T$, and the cost is $\frac{K}{T}+\frac12 hD\,T$, minimized at $T^*=\sqrt{2K/(hD)}$. Same square-root law, balance a $1/T$ term against a linear-in-$T$ term.

So my first move: give item $i$ the full cost of its own order, $K_i = A + a_i$, and let it run its own EOQ, $T_i=\sqrt{2(A+a_i)/(h_iD_i)}$. Done?

No — wait, that's clearly throwing money away. If I do that, every item pays a full $A$ at every one of its orders. Item 1 orders at times $0, T_1, 2T_1,\dots$, item 2 at $0,T_2,2T_2,\dots$, and these cycles are generically incommensurate, so the order epochs essentially never line up. Almost every truck I dispatch carries exactly one item and eats a whole $A$ for it. But the whole point of $A$ is that it's *shared* — one truck could carry items 1, 2, and 5 together and split that single $A$ three ways. Independent EOQ uses none of that. The major cost is an economy of scope and I'm forfeiting it entirely.

So I need the items to *meet*. The cleanest way to make orders coincide is to put everyone on a common rhythm. Pick a basic cycle length $T$, and declare that replenishment opportunities happen only at $0, T, 2T, 3T,\dots$. Now, when do I pay $A$? Once per basic cycle, at rate $1/T$ — and if multiple items happen to order on the same opportunity, that one $A$ covers them all. That's the sharing I wanted.

But I shouldn't force every item to order at *every* opportunity. A slow-moving, expensive-to-hold item ordered as often as a fast cheap one would pile up inventory I'm paying $h_i$ on. So let item $i$ skip: it orders only on every $k_i$-th opportunity, for some positive integer $k_i$. Its cycle becomes $k_iT$, its batch $D_i k_i T$, average inventory $\tfrac12 D_i k_i T$. The integer constraint matters — if $k_i$ could be any real number the items would drift out of phase again and stop sharing. Keeping $k_i$ an integer multiple of a *common* $T$ is exactly what guarantees that whenever two items order, they order at the same instant. That's the design: a basic period $T$ and a vector of integers $\mathbf k=(k_1,\dots,k_M)$.

Let me write the cost of this policy. Major: paid every basic cycle, rate $A/T$. Minor for item $i$: paid every $k_iT$, rate $a_i/(k_iT)$. Holding for item $i$: $\tfrac12 h_i D_i k_i T$. Sum it:

$$ TC(T,\mathbf k)=\frac{A}{T}+\sum_{i=1}^{M}\left[\frac{a_i}{k_iT}+\frac12 h_i k_i D_i T\right]. $$

Let me sanity-check the extremes. One item, $k_1=1$: this is $\frac{A+a_1}{T}+\frac12 h_1 D_1 T$, exactly the EOQ with fixed cost $A+a_1$. Good, it reduces. All $k_i=1$: everyone orders every cycle, $A$ shared by all every time — that's the maximally-coordinated common-cycle policy. So both naive endpoints live inside this family; the $k_i>1$ freedom is what lets slow items skip and lets the policy interpolate between "share everything" and "let each item breathe at its own pace."

Now I have to optimize over $T>0$ and integer $\mathbf k$. Let me group the cost so I can see its shape:

$$ TC(T,\mathbf k)=\frac{1}{T}\Big(A+\sum_i\frac{a_i}{k_i}\Big)+\frac{T}{2}\sum_i k_i h_i D_i. $$

For *fixed* $\mathbf k$ this is again the EOQ shape — call the numerator $\mathcal A(\mathbf k)=A+\sum_i a_i/k_i$ and the coefficient $\mathcal H(\mathbf k)=\sum_i k_i h_i D_i$; it's $\mathcal A/T + \tfrac12\mathcal H\,T$, strictly convex in $T$. Differentiate: $-\mathcal A/T^2 + \tfrac12\mathcal H=0$, so

$$ T^\*(\mathbf k)=\sqrt{\frac{2\mathcal A(\mathbf k)}{\mathcal H(\mathbf k)}}=\sqrt{\frac{2\big(A+\sum_i a_i/k_i\big)}{\sum_i k_i h_i D_i}}. $$

Nice — given the integers, the best basic period is a closed form, one square root. So the genuinely hard part is choosing the integers. The continuous part is free.

How big is the integer space? Each $k_i$ a positive integer, $M$ of them — exponential. I can't sweep it for large $M$. So let me see if I can decouple. Fix $T$ for a moment and look at the cost as a function of $\mathbf k$. The major term $A/T$ doesn't depend on $\mathbf k$ at all. And the rest is a *sum over items* with no cross terms:

$$ TC(T,\mathbf k)=\frac{A}{T}+\sum_i\underbrace{\Big[\frac{a_i}{k_iT}+\frac12 h_i D_i k_i T\Big]}_{f_i(k_i)}. $$

Given $T$, each $f_i$ depends only on its own $k_i$. So for a fixed $T$ the integer problem *separates completely* — I just minimize each item independently. That's a huge collapse: $M$ independent one-dimensional integer problems instead of one $M$-dimensional one.

So minimize $f_i(k)=\frac{a_i}{kT}+\frac12 h_iD_iT\,k$ over positive integers $k$. Relax to real $k$: $f_i'(k)=-\frac{a_i}{k^2T}+\frac12 h_iD_iT=0$ gives $k^2=\frac{2a_i}{h_iD_iT^2}$, so

$$ k_i^{\text{cont}}=\frac{1}{T}\sqrt{\frac{2a_i}{h_iD_i}}. $$

Stare at $\sqrt{2a_i/(h_iD_i)}$ for a second — that's exactly item $i$'s *own* EOQ cycle time $T_i$, the rhythm it would pick if its only fixed cost were its minor cost $a_i$ and there were no sharing. So $k_i^{\text{cont}}=T_i/T$: the ideal multiplier is just the ratio of the item's natural cycle to the basic period. An item that naturally wants to order much less often than the basic period gets a big $k_i$; one that wants the basic rhythm gets $k_i\approx1$. That's a satisfying reading — the integers are telling each item how many basic cycles to wait so that $k_iT$ sits near its own EOQ.

But $k_i$ must be an integer $\ge 1$. $f_i$ is convex in real $k$, so the integer optimum is one of the two integers bracketing $k_i^{\text{cont}}$ (and at least 1). I don't even need to guess which — convexity gives a clean rounding test: start from $\lfloor k_i^{\text{cont}}\rfloor$ (floored at 1) and step toward whichever neighbor lowers $f_i$, i.e. keep increasing while $f_i(k+1)<f_i(k)$ and decreasing while $f_i(k-1)<f_i(k)$. That lands on the integer minimizer. (If I want it in closed form: $f_i(k+1)\ge f_i(k)\iff k(k+1)\ge \frac{2a_i}{h_iD_iT^2}=(T_i/T)^2$, the standard threshold for rounding a convex EOQ multiplier — pick the smallest $k\ge1$ that passes.)

Now I have two cheap maps: given $\mathbf k$, the best $T$; given $T$, the best $\mathbf k$. They're each other's complements, so alternate them. Start somewhere reasonable — $k_i=1$ for all $i$ (the common-cycle guess), compute $T^\*(\mathbf 1)$. Then update each $k_i$ optimally at that $T$. Then recompute $T^\*$ at the new $\mathbf k$. Repeat until $\mathbf k$ stops changing.

Does this terminate, and is it any good? Each half-step weakly decreases $TC$ (each is an exact minimization of one block while the other is held), and the cost is bounded below by zero, so the alternation is monotone and the integer vector lives in a bounded box once $T$ settles, so it converges in a few iterations. It's a coordinate-descent — it finds a *local* optimum, not guaranteed global. Let me worry about that.

Here's the snag — coordinate descent can get stuck. Concretely: I fix $\mathbf k$, get a $T$; I fix that $T$, the per-item rounding might keep $\mathbf k$ where it is even though a different $\mathbf k$ with its *own* matching $T$ would be cheaper, because the rounding only ever looks at the current $T$. The iteration can be trapped by its starting point. So I want either a guarantee or a global sweep.

The global sweep is actually available and cheap, because of a structural fact about $T$. Think of $\mathbf k$ as a function of $T$: as $T$ shrinks, each item's ideal $k_i^{\text{cont}}=T_i/T$ grows, so the optimal integer $k_i(T)$ is a non-decreasing step function of $1/T$. Between the jumps, $\mathbf k$ is constant and $TC(\cdot,\mathbf k)$ is just the smooth EOQ curve, convex in $T$. So $TC(T)=\min_{\mathbf k}TC(T,\mathbf k)$ is piecewise convex, the pieces delimited by the finitely many $T$-values where some $k_i$ flips. And those breakpoints are explicit: $k_i$ flips from $k$ to $k+1$ as $T$ falls through the point where $k(k+1)=(T_i/T)^2$, i.e. $T=T_i/\sqrt{k(k+1)}$. Each item contributes a sequence of breakpoints; on each interval between consecutive breakpoints I know $\mathbf k$ exactly and can drop in the closed-form $T^\*(\mathbf k)$ (clipped to the interval) and read off the cost. Enumerate the intervals, take the best — that's a global optimum over the whole basic-period family, in time linear in the number of breakpoints rather than exponential in $M$. And $T$ itself is bounded: it can't usefully be smaller than the smallest single-item EOQ cycle or larger than the all-together common cycle, which caps how many breakpoints I ever have to visit.

So I have a fast, near-exact recipe: either iterate $T\leftrightarrow\mathbf k$ to a local optimum, or sweep the $T$-breakpoints for the global one. Good. But I'm uneasy about one thing — the integers can come out arbitrary ($1,2,3,5,7,\dots$), and arbitrary integers don't *physically* line up as nicely as I assumed. If item A has $k_A=2$ and item B has $k_B=3$, their orders coincide only every $6$ basic cycles; in between, each orders alone and pays… well, it pays its minor cost, but does it also drag a major cost $A$ onto an opportunity that no other item shares? In the strict basic-period accounting I wrote, $A$ is charged once per basic cycle regardless — every opportunity pays $A$ whether one item or all items show up. That's clean and it's what makes the cost separable, but it slightly over-charges: an opportunity where, say, no item happens to order still "pays" $A$ in the formula. For the moment I'll keep the strict charging (it's the conservative, tractable model and it upper-bounds the true cost); the alternative — only charging $A$ on opportunities where at least one item actually orders — couples the items through the *pattern* of coincidences and destroys the clean separability. I note the gap and move on with the tractable version.

Now the deeper worry: how far from optimal is this whole restriction? I've assumed the optimal policy is "common basic period, integer multiples." The truly optimal continuous-time schedule needn't be cyclic at all, and even among cyclic policies the cycles needn't be integer multiples of a common base. Am I leaving a lot on the table by insisting on this structure?

Let me think about the worst case the structure could cause, using the EOQ curve itself. The cost of one item on a cycle $t$ is $g(t)=c_1/t + c_2 t$ with minimum at $t^\*=\sqrt{c_1/c_2}$ and $g(t^\*)=2\sqrt{c_1c_2}$. If I'm forced to use $t=r\,t^\*$ instead of the optimum, the cost ratio is $\frac{g(rt^\*)}{g(t^\*)}=\frac{c_1/(rt^\*)+c_2 r t^\*}{2\sqrt{c_1c_2}}=\frac12\big(r+\tfrac1r\big)$. This is the key quantity: the EOQ cost is remarkably *flat* near its minimum. Being off by a factor $r$ in the cycle costs only $\tfrac12(r+1/r)-1$, which is quadratic-small for $r$ near 1. Off by $\pm10\%$? $\tfrac12(1.1+1/1.1)\approx1.0045$ — less than half a percent. That flatness is the gift that makes coordinating worthwhile: forcing an item onto a slightly-wrong cycle barely hurts it.

So how much can the *common-base, integer-multiple* restriction cost me in the worst case? Suppose I additionally restrict the multipliers to be **powers of two** — $k_i\in\{1,2,4,8,\dots\}$. Why would I want that extra restriction? Because powers of two are *nested*: any two intervals $2^a T$ and $2^b T$ have one dividing the other, so the order epochs of every pair of items are perfectly aligned — whenever the slower item orders, the faster one is ordering too. No more "they coincide only every 6 cycles" awkwardness; the strict and true charging of $A$ come much closer together, and the whole schedule is a clean dyadic hierarchy. And the search collapses even further: each $k_i$ is just a choice of exponent, the nearest power of two to $k_i^{\text{cont}}$.

What does power-of-two rounding cost? Each item's chosen multiple is within a factor $\sqrt2$ of its continuous ideal: the worst case is when $k_i^{\text{cont}}$ sits exactly at the geometric midpoint between two powers of two, $r\in[1/\sqrt2,\sqrt2]$. Plug the extreme $r=\sqrt2$ into the flatness formula: $\frac12\big(\sqrt2+\frac1{\sqrt2}\big)=\frac12\cdot\frac{3}{\sqrt2}=\frac{3}{2\sqrt2}\approx1.0607$. So with the base period $T$ *fixed*, the worst-case loss from snapping every item to a power of two is about $6\%$. That's the Brown-style single-item bound carried item-by-item: because the cost is flat and powers of two are at most $\sqrt2$ off, no item loses more than $6\%$, so neither does the sum.

And here's the lever that makes it even better: I don't have to fix the base. If I'm free to slide $T$ continuously, I can choose it to center the dyadic grid favorably across all items at once. Intuitively, the $6\%$ was the cost of the single worst-aligned item; by tuning the one shared degree of freedom $T$ I can keep all items closer to the good part of their flatness curves simultaneously, and the worst case drops to about $2\%$. So a power-of-two policy with optimized base period is within roughly $2\%$ of the *true* optimum — not just the best policy of my restricted family, but the unconstrained continuous-time optimum. That's the payoff: the awkward exponential integer problem is replaced by "round each item's ideal multiple to the nearest power of two and tune one $T$," and it's provably near-optimal. The restriction I worried was lossy is the restriction that makes the problem both tractable *and* tight.

Let me also reconcile this with the alternative I had on the table — the event-driven can-order idea, where each item has a must-order point and a can-order point, and any item hitting must-order triggers a setup that also tops up every item below its can-order point. That's the right shape when demand is *random* and I want coordination to be opportunistic. But for steady, known demand it's heavier than it needs to be: I'd be optimizing three thresholds per item with messy interactions, and with deterministic demand the order epochs are predictable anyway — so a fixed cyclic schedule captures the same sharing with far fewer parameters and an actual optimality guarantee. I'll keep the can-order policy in mind for the stochastic case, but for the deterministic problem the basic-period / integer-multiple policy is the one to land on.

So the method is: parameterize by a basic period $T$ and integer multipliers $k_i$; cost is $TC(T,\mathbf k)=\frac1T(A+\sum_i a_i/k_i)+\frac T2\sum_i k_i h_iD_i$; for fixed $\mathbf k$ the best $T$ is the closed-form square root; for fixed $T$ the best integer $k_i$ is the convex-rounded $T_i/T$ per item; alternate to a local optimum or sweep the $T$-breakpoints for the global one; and if I want guaranteed near-optimality with the simplest schedule, restrict the $k_i$ to powers of two and tune the single base period.

Now the code. I'll write the cost, the two closed-form blocks, the iteration, the power-of-two variant, and a brute-force enumerator to check a small instance.

```python
import math, itertools

def jrp_cost(T, k, A, a, h, D):
    # major shared once per basic cycle; minor per item every k_i*T; holding on D_i*k_i*T/2
    major = A / T
    minor = sum(a[i] / (k[i] * T) for i in range(len(k)))
    hold  = (T / 2.0) * sum(k[i] * h[i] * D[i] for i in range(len(k)))
    return major + minor + hold

def optimal_T(k, A, a, h, D):
    # fixed k -> EOQ-shape in T: T* = sqrt(2(A + sum a_i/k_i) / sum k_i h_i D_i)
    num = 2.0 * (A + sum(a[i] / k[i] for i in range(len(k))))
    den = sum(k[i] * h[i] * D[i] for i in range(len(k)))
    return math.sqrt(num / den)

def best_k_given_T(T, a, h, D):
    # fixed T -> cost separates over items; minimize each convex f_i(k)=a_i/(kT)+0.5 h_i D_i T k
    k = []
    for i in range(len(a)):
        kc = math.sqrt(2.0 * a[i] / (h[i] * D[i] * T * T)) if a[i] > 0 else 0.0  # = T_i^EOQ / T
        kk = max(1, int(math.floor(kc)))
        f = lambda kv: a[i] / (kv * T) + 0.5 * h[i] * D[i] * kv * T
        while f(kk + 1) < f(kk):      # convex: step toward the cheaper neighbor
            kk += 1
        while kk > 1 and f(kk - 1) < f(kk):
            kk -= 1
        k.append(kk)
    return k

def iterative_jrp(A, a, h, D, max_iter=100):
    # coordinate descent: alternate T <- best(k) and k <- best(T)
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
    # restrict k_i to {1,2,4,...}: nested intervals, near-optimal, simplest schedule
    def to_pow2(x):
        if x < 1: return 1
        lo = 2 ** int(math.floor(math.log2(x)))
        return lo if (x / lo) <= ((2 * lo) / x) else 2 * lo  # nearest in log scale
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
    # exact reference: enumerate integer multipliers, take closed-form T for each
    best = None
    for k in itertools.product(range(1, kmax + 1), repeat=len(a)):
        T = optimal_T(list(k), A, a, h, D)
        c = jrp_cost(T, list(k), A, a, h, D)
        if best is None or c < best[0]:
            best = (c, T, list(k))
    return best
```

Checking it on a small instance — $A=100$, $a=(10,20,5)$, $h=(1,0.5,2)$, $D=(100,50,200)$ — the iteration lands on $\mathbf k=(1,2,1)$ with $T\approx0.674$ and cost $\approx370.8$, the power-of-two variant lands on the *same* $(1,2,1)$ and cost, and the brute-force enumeration confirms that's the true minimum. And a single item with no minor cost gives back $T^\*=\sqrt{2K/(hD)}$ exactly — the EOQ I started from. So the whole chain closes: independent EOQ wastes the shared $A$; putting items on a common basic period with integer multiples lets them share it; the cost is EOQ-shaped in $T$ for fixed multiples (closed-form $T^\*$) and separable in the multiples for fixed $T$ (per-item convex rounding of $T_i/T$); alternating the two converges, sweeping the $T$-breakpoints globalizes it, and snapping the multiples to powers of two with a tuned base period buys a clean nested schedule that's provably within a couple percent of the true optimum.
