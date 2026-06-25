The inscribed eleven-gon gave me `0.0215`, a bit over half the record, and the lesson from it was
blunt: no closed-form configuration I can name reproduces the irregular `1/27` optimum, so I have to
start *searching* — actually trying configurations and letting the evaluator pick the best. The very
simplest search, the one that needs no temperature, no gradient, no schedule, is to throw darts:
sample many point sets uniformly at random in the square, score each one exactly, and keep the best
minimum-area I ever see. It is brute force, but it is honest, it is trivially correct, and it gives
me a clean reading of how much the *raw landscape* offers before I invest in anything cleverer.

The appeal is that it is embarrassingly parallel and needs nothing from me but a sample count.
Every trial is independent: draw eleven points uniformly in `[0,1]^2`, compute the minimum over all
`165` triangles, and if that minimum beats my running best, store the configuration. There is no
state, no acceptance rule, no way to get stuck — I just keep the maximum over however many samples I
can afford. So the only real decision is the budget, and I should make it large.

Before I trust any of this, I want the scoring arithmetic itself nailed down, because the whole
method rests on computing a triangle area correctly. The area of the triangle on `a,b,c` is half the
absolute value of the cross product `(b-a) x (c-a)`. Take the unit right triangle `(0,0),(1,0),(0,1)`:
the cross product is `(1)(1) - (0)(0) = 1`, half of it is `0.5`, which is the area I know that
triangle has. Now a near-collinear trio, `(0,0),(1,0),(0.5,0.001)`: the cross product is
`(1)(0.001) - (0.5)(0) = 0.001`, area `0.0005` — a thin sliver, exactly as a near-flat triple should
read. So the per-triple formula is right, and the score of a configuration is just the minimum of
this over all `165` triples. One more sanity check on the *min-of-many* part: the four corners of the
unit square give four triples, each a right triangle with legs `1`, so every triple has area `0.5`
and the minimum is `0.5` — no sliver, because no three corners are collinear. That is the structure
I am fighting: a configuration scores well only if *every* one of its many triples avoids the sliver
regime.

But I have to think about cost, because the naive way is too slow. The exact evaluator loops over
`165` triples per configuration, and if I want millions of configurations that is hundreds of
millions of small area computations in a pure loop — minutes wasted on Python overhead. The fix is
to vectorize: precompute the `165` index triples once, and for a whole *batch* of random
configurations compute all `165` cross-products at once with array arithmetic, take the per-config
minimum, and read off the batch maximum. A batch of tens of thousands of configurations becomes a
handful of array operations. That lets me push the sample count into the millions and still finish
in well under a minute, which is the right scale for a baseline I want to trust.

Now the honest question: how good is random multi-start actually going to be? My worry is the curse
of dimensionality. A configuration of eleven points is a point in twenty-two dimensions, and "good"
configurations — the ones with a fat minimum triangle — ought to occupy a vanishingly small corner
of that space, because with eleven points scattered at random it is very likely that *some* trio
happens to be nearly collinear, and that sliver, being the minimum, sets the score. But "ought to"
is a guess; let me measure it. I draw `200,000` uniform configurations and look at the distribution
of their min-areas. The mean min-area comes out to `0.00059` and the median to `0.00038` — so a
*typical* random configuration scores under one part in sixty of the `1/27` record, dominated by its
worst sliver, which is the curse made concrete. The 99th percentile is only `0.0030`. So the bulk of
the distribution is junk, and multi-start lives entirely in its extreme right tail: out of those
`200,000` draws the single best min-area is `0.0102`, about `0.275` of the record. That is a factor
of seventeen above the mean — the tail is fat enough that volume buys a lot — but it is already only
a quarter of the way to the record, drawn from two hundred thousand samples.

That number, `0.0102` from `200k`, is the one I should extrapolate from, and it tells me to expect
sharply diminishing returns rather than a steady climb. I check the growth directly on a million
draws: best-of-`10,000` is `0.0068` (`0.18` of record), best-of-`100,000` is `0.0096` (`0.26`),
best-of-`1,000,000` is `0.0111` (`0.30`). So each tenfold increase in budget moves the best score by
less than the previous tenfold did — from `0.18` to `0.26` to `0.30` of record — exactly the
flattening I would expect from sampling the tail of a fixed distribution: to keep gaining I would
have to reach ever-rarer draws, and the rarity grows far faster than the payoff. The curve is
visibly bending toward a ceiling somewhere just above `0.30` of record. There is no mechanism here
to *improve* a promising configuration; the method can only report the luckiest draw, and luck on
this scale has steeply diminishing reach.

So I set the budget where the curve has mostly flattened but a little headroom remains —
`4,000,000` draws in batches of `50,000`, fixed seed for reproducibility — and run it. The result is
`0.01087`, which is `0.294` of the record, and it finishes in about `48` seconds. That sits right on
the extrapolated curve: `4M` lands between the `1M` reading of `0.0111` and no further meaningful
gain, and the whole jump from `1M` to `4M` is essentially noise. So multi-start clears the noise
floor by a wide margin — seventeen-fold above the average draw — but stalls at under a third of the
record, well short of even the `0.0215` eleven-gon, and the diminishing-returns curve says throwing
more samples at it will not change that. That is the ceiling of pure sampling on this landscape.

The reason for the ceiling is also the opening for the next rung. Random multi-start wastes all its
information: it finds a decent configuration and then immediately forgets it and draws a fresh random
one. Every good draw is thrown away the instant the next is scored; nothing carries forward. What I
want instead is to *hold onto* a good configuration and nudge it — move one point a little, see if
the minimum triangle got bigger, and if so keep the nudge. That is local search, and the danger with
naive local search (greedy hill-climbing) is that fixing one bad triangle by moving a point usually
flattens some *other* triangle, so the minimum bounces around and a purely greedy climber gets stuck
in a shallow trap almost immediately. The cure is to accept some worsening moves on purpose, with a
probability that cools over time — simulated annealing — so the search can walk out of shallow traps
early and settle into a deep basin late. So the next rung keeps the random *starting* points but adds
the thing this rung lacks: memory plus a principled rule for moving one point at a time toward a
larger minimum triangle.
