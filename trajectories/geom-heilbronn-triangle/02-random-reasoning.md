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

But I have to think about cost, because the naive way is too slow. The exact evaluator loops over
`165` triples per configuration, and if I want millions of configurations that is hundreds of
millions of small area computations in a pure loop — minutes wasted on Python overhead. The fix is
to vectorize: precompute the `165` index triples once, and for a whole *batch* of random
configurations compute all `165` cross-products at once with array arithmetic, take the per-config
minimum, and read off the batch maximum. A batch of tens of thousands of configurations becomes a
handful of array operations. That lets me push the sample count into the millions and still finish
in well under a minute, which is the right scale for a baseline I want to trust.

Now the honest question: how good do I *expect* random multi-start to be? My worry is the curse of
dimensionality. A configuration of eleven points is a point in twenty-two dimensions, and "good"
configurations — the ones with a fat minimum triangle — occupy a vanishingly small corner of that
space. A uniform random configuration is almost always *bad* for the minimum, because with eleven
points scattered at random it is very likely that *some* trio happens to be nearly collinear: three
points that nearly line up make a sliver, and that sliver, being the minimum, sets the score. The
expected smallest triangle among random points is tiny — it falls off fast as `n` grows, because
the number of triples grows like `n^3` and the worst of many independent slivers is very thin. So
random sampling spends almost all its trials on configurations whose score is a fraction of a
percent of the record, and only occasionally stumbles onto one where, by luck, no trio is too
collinear. The best of millions of such lucky draws will be far better than the average draw, but I
do not expect it to be *good* in absolute terms — the probability of randomly landing near the
structured `1/27` optimum, with its points carefully placed on boundary and lattice, is essentially
nil.

So my prediction is that random multi-start clears the noise floor but stalls low — somewhere in the
neighborhood of a third of the record, maybe `0.010`–`0.012`, bought by sheer volume of samples. It
will beat the *average* random draw by a wide margin and it will confirm that the landscape has
better-than-typical configurations lying around, but it will not approach the eleven-gon, let alone
the record, because it has no mechanism to *improve* a promising configuration — it can only report
the best one it happened to draw. Every good draw is thrown away the instant the next draw is
scored; nothing carries forward.

That last point is the whole opening for the next rung, and I can already see it. Random multi-start
wastes all its information: it finds a decent configuration and then immediately forgets it and
draws a fresh random one. What I want instead is to *hold onto* a good configuration and nudge it —
move one point a little, see if the minimum triangle got bigger, and if so keep the nudge. That is
local search, and the danger with naive local search (greedy hill-climbing) is that fixing one bad
triangle by moving a point usually flattens some *other* triangle, so the minimum bounces around and
a purely greedy climber gets stuck in a shallow trap almost immediately. The cure is to accept some
worsening moves on purpose, with a probability that cools over time — simulated annealing — so the
search can walk out of shallow traps early and settle into a deep basin late. So the next rung keeps
the random *starting* points but adds the thing this rung lacks: memory plus a principled rule for
moving one point at a time toward a larger minimum triangle.
