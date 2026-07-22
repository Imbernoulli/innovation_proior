The previous rung told me precisely what was wrong with the baseline: a fixed lexicographic order
is correct but geometry-blind, and it lands on exactly `2^n`. I ran the lexicographic greedy out
to `n = 6` to be sure that floor is really `2^n` and not just a small-`n` coincidence — it returns
`2, 4, 8, 16, 32, 64` for `n = 1..6`, dead on `2^n` every time. So the order really is pinning the
output: the construction commits to whatever the counting order hands it next, with no global choice
about where to spend the cap's budget, and the counting order happens to funnel it into a power of
two.

If the order is the lever, the cheapest thing I can do is stop trusting any one order. The admission
rule — add a point iff it does not close a line with an already-admitted point, and on admission
block the completing point of every line it forms — is what guarantees validity, and it has nothing
to do with the order in which points are offered. So I can permute the offer order freely and still
get a valid cap by construction; only the *size* changes. The question is whether changing the order
actually buys anything, and that I cannot settle by assertion — different orders might all collapse
to `2^n` for some structural reason, the way lexicographic does. I need to look at the actual spread
of sizes a random order produces.

Let me take `n = 4`, where the proven optimum is `20` and the floor is `16`, and just measure: draw
5000 uniformly random offer orders, run the greedy fill on each, and tabulate the resulting cap
sizes. The outcome:

```
size 16:  477   ( 9.5%)
size 17: 3032   (60.6%)
size 18: 1488   (29.8%)
size 20:    3   ( 0.1%)
mean = 17.20,  min = 16,  max = 20
```

Several things in this table are worth dwelling on. First, the modal random order gives `17`, and
the mean is `17.2` — so a *typical* random order already beats the lexicographic `16`, but only
barely, by about one point. That kills any hope that a single random order is reliably good; the
average run is essentially as mediocre as lexicographic. Second, and this is the part I did not
expect: the distribution jumps straight from `18` to `20` with nothing at `19`. The greedy fill at
`n = 4` apparently lands on `19` essentially never — the sizes it produces are quantized, and `20`
is reachable but rare, showing up in only 3 of 5000 runs. So the value of this idea is not in the
typical order at all; it is entirely in the right tail, and the right tail is thin.

That reframes what the method is. Taking one random order is pointless — its expectation is `17.2`,
a marginal gain. The operative move is the *maximum* over many independent orders: best-of-`k`, keep
the largest cap seen and throw the rest away. Best-of-sample reaches into the right tail in a way
that any single sample cannot. To see how deep I have to reach, I tracked the running maximum over
the 5000 `n = 4` runs: it first hits `18` at restart `8`, but does not reach `20` until restart
`361`. So a handful of restarts clears the easy gain to `18`, but it takes a few hundred to catch
one of the rare `20`-orders. Crucially, `20` *is* reachable — multi-start at `n = 4` finds the
proven optimum, which is the first time on this ladder a pure-sampling construction matches a known
optimal value. I verified the returned `20`-cap with the full incremental checker and it passes.

So at `n = 4` the recipe is: same admission rule, uniformly random offer order, take the best over
several thousand restarts. The next question is how this scales, and I should be wary of extrapolating
the `n = 4` success — the optimum there sits only `4` above the floor, an easy reach. Let me look at
`n = 5`, optimum `45`, floor `32`. Over 2000 random orders the size distribution has mean `35.6`,
median `36`, 99th percentile `38`, and best `38`. The typical order again beats the floor by a few
points, and the tail tops out around `38`. To squeeze past that I tracked best-of-`k`:

```
k = 10:   37
k = 50:   37
k = 100:  38
k = 500:  38
k = 1000: 39
k = 5000: 39
```
