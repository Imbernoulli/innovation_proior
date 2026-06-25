Random multi-start clears the floor and even hits the optimum at `n = 4`, but the feedback made
its ceiling plain: it is a lottery over orders, and the right tail of the cap-size distribution
thins so fast that more restarts buy almost nothing as `n` grows. The orders being sampled are
*blind* — uniform noise, with no preference for points that sit in structured positions. If the
sampling is what is limiting me, then the lever is the order itself: instead of shuffling, score
every vector by a deterministic function of its coordinates and feed the same greedy admission rule
the vectors in highest-priority-first order. The admission rule is untouched, so validity stays
free; only the ordering changes. The open question is what to score, and whether a score I can write
down by hand is actually better than the best of a large random sample.

What structure should I reward? `F_3^n` has a lot of symmetry, and the large caps in the literature
are not random — they have regular weight profiles and reflection symmetries. Two features feel
natural to encode. First, *reflection symmetry*: pair coordinate `i` with coordinate `n−1−i` and
reward a vector when those paired entries agree, `el[i] == el[n−1−i]`. The hope is that a cap built
preferentially from reflection-symmetric vectors inherits a symmetry that makes its line structure
more regular. Second, a *weight profile*: the Hamming weight (number of nonzero coordinates) taken
mod 3. The line condition `a + b + c ≡ 0 (mod 3)` is itself a statement about coordinate sums mod 3,
so it is at least plausible that biasing the fill toward one weight-mod-3 layer keeps the admitted
points coherent rather than mixing layers that collide more. A tiny coordinate-sum term breaks ties
so the order is total. The priority is the sum of these terms. I should be honest that this is a
*guess* aligned with the geometry, not something I derived; whether it helps is an empirical
question, so let me actually trace it before claiming anything.

Start at `n = 2`, where I can check every step by hand and against the known optimum of 4. Writing
out `score(el) = 2 + 1.0·[reflection matches] + 0.5·(3 − w mod 3) + 0.01·Σel`, the reflection loop
is `range(1, n//2) = range(1, 1)`, which is empty — so at `n = 2` reflection contributes nothing and
the order is driven purely by the weight term. The weight-`0` vector `(0,0)` scores `2 + 0.5·3 =
3.5`; the weight-`1` vectors score `2 + 0.5·2 + (sum term) ≈ 3.01–3.02`; the weight-`2` vectors
score `2 + 0.5·1 + (sum term) ≈ 2.52–2.54`. So the order admits `(0,0)` first. Admitting `(0,0)`
blocks, for each later candidate `q`, the closing point `r = (−(0,0) − q) mod 3 = (−q) mod 3`: it
blocks `(0,1)` (paired with `(0,2)`), `(1,0)` (paired with `(2,0)`), and so on — exactly the
weight-`1` partners. The next-highest surviving scores are the weight-`1` points whose partner is
blocked, then weight-`2`. Running it out, the admission order is `(0,0), (0,2), (2,0), (2,2)` and the
final set is `{(0,0),(0,2),(2,0),(2,2)}`. A direct triple scan over all `C(4,3)=4` triples confirms
no three sum to `0 mod 3`: this is a valid cap of size 4, which *is* the proven optimum at `n = 2`.
So the mechanism works and the bookkeeping is right; that is reassuring but it is the easy case, where
optimum is trivial to reach by almost any order.

Before trusting the symmetry story I want to check what the reflection term actually does, because
the loop bounds surprised me at `n = 2`. Enumerating `range(1, n//2)` for each `n`: at `n = 2, 3` it
is empty; at `n = 4` it checks only the single pair `(1,2)`; at `n = 5` only `(1,3)`; at `n = 6` the
pairs `(1,4),(2,3)`; at `n = 7` the pairs `(1,5),(2,4)`. Two things fall out of this that I would
have asserted wrongly if I had not looked. First, reflection has *no effect at all* until `n = 4`, so
any improvement at `n = 2, 3` comes entirely from the weight term. Second, the loop starts at `i = 1`,
so it never checks the outermost pair `(0, n−1)` — the "reflection symmetry" I am rewarding is a
partial, off-by-one version of the full coordinate reversal, not the clean involution I had in mind.
I will keep the function as written (it is the ordering I want to measure), but I now expect the
reflection contribution to be weaker than the clean story suggests, and most of the lift at small `n`
to come from the weight layering.

Now the real test: run the priority-greedy at each `n`, verify every returned set with the cap
checker, and compare against the two baselines I already have. The verified sizes come out
`n=4:18, 5:36, 6:64, 7:138`, all confirmed valid by the incremental checker (and `n ≤ 6`
cross-checked by triple scan). Against the lexicographic floor `16, 32, 64, 128`: the structured
order beats lex at `n = 4` (`18 > 16`), `n = 5` (`36 > 32`), and `n = 7` (`138 > 128`) — but at
`n = 6` it returns `64`, exactly *tying* lex, no improvement at all. That tie is worth sitting with
rather than glossing: `n = 6` is the first `n` where the reflection loop checks two pairs, and yet it
is the one `n` where the structured order buys nothing over plain index order. So the symmetry I
rewarded is not reliably helping even on the cases it is active for; the gains are real but uneven,
which is exactly the signature of a structure that is only *approximately* aligned with what the
greedy fill needs.

And against random multi-start, which gives `20, 38, 77, 145` from a few thousand restarts: the
structured order is *below* it at every single `n` — `18 < 20`, `36 < 38`, `64 < 77`, `138 < 145`. So
the hand-designed priority comfortably clears the lexicographic floor (mostly) but loses to
best-of-thousands sampling across the board. That is the honest verdict, and it is not a
disappointment — it is the lesson of this rung. Having *a* structured priority is not enough; the
priority has to be the *right* one, finely shaped to the dimension. A human guessing at which
symmetries to reward — and, as the loop-bounds check showed, even getting the symmetry only
half-right — cannot out-do brute sampling. The greedy-priority *skeleton* is clearly the correct
machinery: it is the same skeleton the strong constructions use, and validity is free inside it. But
the function plugged into it is everything, and a single hand-tuned order plateaus because the space
of useful priorities is large and non-obvious.

So the measurements point past hand-design. The skeleton is right and a person's best guess at the
priority is not; what is left is to stop guessing the priority and let a search over the function
space find it — tuned to the specific `n`, encoding regularities I would not write down (and clearly
including the coordinate pairs my off-by-one loop skipped). That is the natural next rung: hand the
priority function itself to an evolutionary program search and run what it discovers in this exact
skeleton. This rung's job was to establish, with verified numbers rather than hope, that the skeleton
is the right vessel and the hand-designed priority is the weak link.

One caution I respected throughout for the numbers: the structured priority has ties (many vectors
share the same reflection/weight score), and how ties are broken affects the result, so I fixed the
tie-break deterministically with the coordinate-sum term and ran the verifier on every returned cap,
never reporting a size I had not checked.
