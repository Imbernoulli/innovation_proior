# Knaster-Tarski Fixed-Point Theorem

Let `(L, <=)` be a complete lattice, and let `f : L -> L` be monotone:

```text
x <= y  implies  f(x) <= f(y).
```

Then the set

```text
Fix(f) = {x in L | f(x) = x}
```

is a complete lattice under the inherited order. In particular, `f` has a least fixed point and a greatest fixed point.

The extremal fixed points are:

```text
lfp(f) = meet {x in L | f(x) <= x}
gfp(f) = join {x in L | x <= f(x)}
```

The first set consists of pre-fixed points, or upper stable bounds. The second consists of post-fixed points, or lower stable bounds.

## Proof

Let

```text
P = {x in L | f(x) <= x}.
```

Since `top in P`, the set is nonempty. Put `a = meet P`. For every `p in P`, `a <= p`, so by monotonicity `f(a) <= f(p) <= p`. Thus `f(a)` is a lower bound of `P`, and therefore `f(a) <= a`. This means `a in P`; since `a` is the meet of `P`, `a <= f(a)`. Hence `f(a) = a`. Every fixed point belongs to `P`, so `a` is below every fixed point. Thus `a = lfp(f)`.

Dually, let

```text
Q = {x in L | x <= f(x)}.
```

Since `bot in Q`, the set is nonempty. Put `b = join Q`. For every `q in Q`, `q <= b`, so `f(q) <= f(b)`, and therefore `q <= f(b)`. Thus `f(b)` is an upper bound of `Q`, so `b <= f(b)`. This means `b in Q`; since `b` is the join of `Q`, `f(b) <= b`. Hence `f(b) = b`. Every fixed point belongs to `Q`, so every fixed point is below `b`. Thus `b = gfp(f)`.

It remains to show that `Fix(f)` is complete. Let `A subseteq Fix(f)`, and let `s = join_L A`. For each `x in A`, `x = f(x) <= f(s)`, hence `s <= f(s)`. Therefore the interval `[s, top]` is closed under `f`: if `s <= y <= top`, then `s <= f(s) <= f(y) <= top`. This interval is itself a complete lattice, so the least-fixed-point argument applied inside `[s, top]` gives a fixed point `j >= s`. It is the least fixed point above `s`, hence the least fixed-point upper bound of `A`; therefore `j = join_{Fix(f)} A`.

The meet construction is dual. If `t = meet_L A`, then `f(t) <= t`, the interval `[bot, t]` is closed under `f`, and the greatest-fixed-point argument inside that interval gives the greatest fixed point below `t`, which is `meet_{Fix(f)} A`.

Thus arbitrary joins and meets exist inside `Fix(f)`, so `Fix(f)` is a complete lattice.

The proof uses only order: complete-lattice joins and meets provide global bounds, and monotonicity keeps those bounds stable until the one-sided inequalities collapse to equality. No contraction metric, topology, compactness, convexity, or continuity is required.
