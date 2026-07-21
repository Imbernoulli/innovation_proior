**The word that defines this problem is "non-empty".** The operation set is ordinary — point-set `1 i x`, range-query `2 l r` — but the query wants the maximum sum of a *non-empty* contiguous subarray inside `a[l..r]`, and that clause is where the whole difficulty sits. It removes the "take nothing, score 0" escape hatch: an all-negative range must return its least-bad single element (a negative number), an all-zero range returns `0`, and only a range containing a positive returns a positive. Everything hard here is the sign bookkeeping around that rule.

Before any algorithm the scale fixes the arithmetic. With `n, q <= 2*10^5` and `|a[i]| <= 10^9`, a range *total* can reach `2*10^5 * 10^9 = 2*10^14`, four orders of magnitude past the 32-bit ceiling of `~2.1*10^9`. So every stored field and every accumulator has to be `long long`; an `int` anywhere is a silent wrong-answer that only shows on the large tests.

**Why the direct approach can't ship.** The static maximum subarray is Kadane in `O(n)`. The lazy plan — keep `a` plain, apply `1 i x` in `O(1)`, answer each `2 l r` by a fresh Kadane scan over `a[l..r]` — is obviously correct and makes a fine brute-force oracle, but it is `O(nq)`, up to `4*10^10` element touches, hopeless under 2 seconds. I need point updates *and* arbitrary-range queries each in `O(log n)`, which is a segment tree.

**Node summary and merge.** The standard device for "best subarray under merges" makes each node summarize its range with four longs, chosen so two adjacent summaries fuse in `O(1)` without touching the underlying elements:

- `tot`  = sum of the whole range (used to thread a prefix/suffix through a child),
- `pre`  = best non-empty prefix sum (subarray starting at the left end),
- `suf`  = best non-empty suffix sum (subarray ending at the right end),
- `best` = best non-empty subarray sum anywhere in the range.

Merging left `L` and right `R` into parent `P`:

- `P.tot  = L.tot + R.tot`.
- `P.pre  = max(L.pre, L.tot + R.pre)` — the best prefix of `P` stays inside `L`, or spans all of `L` and continues into a best prefix of `R`.
- `P.suf  = max(R.suf, R.tot + L.suf)` — symmetric, anchored at the right end.
- `P.best = max(L.best, R.best, L.suf + R.pre)` — the best subarray lies entirely in `L`, entirely in `R`, or straddles the boundary as (best suffix of `L`) + (best prefix of `R`).

A leaf over one value `v` is `{v, v, v, v}`: the only non-empty subarray is the element itself. Build is `O(n)`, update and query `O(log n)`. The high-level idea isn't the risk; the risk is entirely in the base cases the negatives-and-zeros corner is built to expose — the identity used for out-of-range pieces, and the sign of a lone negative leaf.

**The identity element — the real trap.** A range query splits at internal nodes, and on branches that fall outside `[l,r]` the recursion has to return *something* that merges harmlessly with real nodes. The tempting choice is the zero node `{0,0,0,0}` — "an empty range contributes nothing." It is wrong, and it is wrong on exactly the corner this problem stresses.

Trace `a = [-5,-2,-8]`, query `2 0 2`, true answer `-2` (every subarray sum is negative; the least-bad is the single `-2`). Somewhere the recursion merges a real leaf `L = {-2,-2,-2,-2}` with an out-of-range `R = {0,0,0,0}`:

`P.best = max(L.best=-2, R.best=0, L.suf + R.pre = -2 + 0 = -2) = 0`.

The `0` from the identity's `best`/`pre` is a *phantom empty subarray* — there is no non-empty subarray summing to `0` anywhere in `[-5,-2,-8]` — and on an all-negative range that phantom beats every real candidate. The empty piece does genuinely sum nothing when threaded through a spanning prefix or suffix, so its `tot` must be `0`; but it can never *host* a non-empty subarray, so `pre`, `suf`, `best` must be a sentinel so negative it can never win a max:

```
Node identity() { return Node{NEG, NEG, 0, NEG}; }
```

Re-trace the same merge with `R = {NEG, NEG, 0, NEG}`: `P.tot = -2`; `P.pre = max(-2, -2 + NEG) = -2`; `P.suf = max(NEG, 0 + (-2)) = -2`; `P.best = max(-2, NEG, -2 + NEG) = -2`. `P` equals the real `[-2]` summary — merging with identity is now a true no-op, and the all-negative range returns `-2`.

**Why `NEG = LLONG_MIN/4` rather than `LLONG_MIN`.** The merge *adds* to possibly-sentinel quantities: `L.tot + R.pre`, `R.tot + L.suf`, `L.suf + R.pre`. A real `tot`/`pre`/`suf` is bounded by `2*10^14` in magnitude, so the worst case is `NEG + (~ -2*10^14)`. With `NEG = LLONG_MIN/4 ~ -2.3*10^18` that lands near `-2.3*10^18`, comfortably above `LLONG_MIN ~ -9.2*10^18` — no underflow. Using `LLONG_MIN` itself, `LLONG_MIN + (negative)` wraps to a large *positive* that would then win `best` — a second, quieter sign bug. The `/4` is the headroom that keeps every `NEG + real` safely negative.

**The lone leaf keeps its sign.** The same "let `0` mean took-nothing" mistake can hide at the leaf: clamping to `{max(v,0), max(v,0), v, max(v,0)}` would make `[-7]` under query `2 0 0` report `0` instead of `-7`. Non-empty is mandatory, so the single-element subarray keeps its sign — the leaf is the unclamped `{v,v,v,v}`, full stop.

Having banned `0` as a default, the sentinel must not also swallow a *genuine* zero element. On `a = [-4,0,-1]`, query `2 0 2` should return `0` (the subarray `[0]`). The element `0` produces a legitimate `{0,0,0,0}` leaf; merging `[-4]` with `[0]` gives `best = max(-4, 0, -4+0) = 0`, and folding in `[-1]` leaves `best = 0`. So a zero *element* wins its max honestly, while "nothing at all" — the identity — never does. That distinction between an element that happens to be zero and no element at all is exactly what the two different nodes encode.

**Recursion and I/O.** Standard segment-tree query: disjoint from `[l,r]` returns `identity()`, fully covered returns the stored node, otherwise recurse and merge. One cleanup — if the range lies entirely in the left child (`r <= mid`) return only the left recursion, symmetrically for the right — so I merge two real partial nodes only on a true straddle. Not required now that identity is a real no-op, but it keeps the reliance on the sentinel minimal. Update is the textbook point update: descend to the leaf, set `{val,...}`, recombine upward. Type-2 answers I accumulate into one output string and write once, since `q = 2*10^5` type-2 queries would otherwise pay per-line flush overhead.

**End-to-end on the sample, including an update.** `a = [-2,1,-3,4,-1,2]`. `2 0 5` → `5` (`[4,-1,2]`). `2 0 2` → `1` (the lone `1`). `1 2 5` sets `a[2]=5`, giving `[-2,1,5,4,-1,2]`; recombination rebuilds every ancestor, no stale state. `2 0 5` → `[1,5,4,-1,2] = 11` (extending through the `-1` to `2` adds `+1` over `[1,5,4]=10`). `2 0 0` → `-2`. Output `5 / 1 / 11 / -2`. I cross-check the whole thing against the plain Kadane brute force on random inputs skewed toward negatives and zeros — the corner the identity fix exists for.

I assemble `build`, `update`, `query`, and a `merge`/`identity`/`leaf` around the `Node` struct into one self-contained C++17 file reading stdin and writing the buffered output.
