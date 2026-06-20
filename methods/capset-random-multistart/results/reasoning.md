The previous rung told me precisely what was wrong: a fixed lexicographic order is correct but
geometry-blind, and it lands on `2^n` exactly because it commits to whatever the counting order
hands it next, with no global choice about where to spend the cap's budget. The order is the
lever, and the cheapest possible way to pull it is to stop trusting any single order. If one
arbitrary order gives `2^n`, then many *different* arbitrary orders will give a spread of cap
sizes — some unlucky, some lucky — and I can simply keep the best one I see. This is randomized
greedy with multi-start, and it costs me nothing conceptually: the admission rule is identical
(add a point if it does not close a line with an existing point), so every run still produces a
valid cap by construction; I only shuffle the order in which points are offered.

Let me reason about *why* a random order should beat lexicographic, and by how much. Lexicographic
order is pathological in a specific way: it grabs a dense cluster of low-index points first, and
that cluster's lines block out a structured swath of the remaining space, funnelling the greedy
process into the rigid power-of-two pattern. A uniformly random order has no such bias — the early
points it grabs are scattered across `F_3^n`, so the lines they generate are scattered too, and the
greedy fill tends to pack more loosely-correlated points before it runs out of room. Intuitively, a
random order breaks the accidental alignment between the enumeration and the blocking structure, so
on average it should clear the `2^n` floor. The catch is variance: any single random order is just
as arbitrary as lexicographic; what makes this work is taking the *maximum* over many independent
orders, so I get the best of a sample rather than the typical one. The more starts I run, the
further into the right tail of the size distribution I reach.

Now, how good can this get? I should be honest about the regime. At small `n` the space is tiny and
the number of distinct greedy outcomes is limited, so a few thousand restarts will quite plausibly
*hit the optimum*. At `n = 4` the optimum is `20`, only modestly above the `16` floor, and `20`-caps
are not rare among random orders — I expect multi-start to find `20` and thus match the proven
optimum there. That would be a real milestone for the ladder: the first rung to reach a known
optimal value, bought purely by sampling orders. As `n` grows, though, the optimum pulls away from
what random greedy can reach. The cap-size distribution under random greedy concentrates well below
the optimum, and the right tail thins out fast, so even with many restarts the best-of-sample lags
the optimum by an increasing margin. At `n = 5` I expect to clear the `32` floor comfortably and
land in the high `30s`, short of `45`. At `n = 6` I expect to roughly clear `64` and reach the
mid-`70s`, far short of `112`. At `n = 7` I expect to roughly double the `128` floor's neighborhood
into the `140s`, far short of `236`. The pattern I anticipate is: a clean win over lexicographic
everywhere, an exact optimum at `n = 4`, and a gap to the optimum that *grows* with `n` — because
unstructured random search has no way to exploit the algebraic regularities that the large caps are
built from.

That growing gap is the real lesson I am setting up. Random multi-start fixes the *bias* of a fixed
order, but it does nothing about the *blindness*: every order it tries is still uniform noise, with
no preference for points that sit in symmetric or structured positions, no preference for points
that block few future candidates over points that block many. It is a lottery over orders, and a
lottery cannot learn the geometry. To go further I cannot just sample more orders — the tail is too
thin — I have to *bias* the order intelligently, scoring each candidate vector by some structured
priority that reflects the symmetry of `F_3^n` and feeds the same greedy admission rule a smarter
sequence. That is the move from "random order" to "priority-function greedy," and it is the next
rung.

One implementation point I want to get right so the numbers are trustworthy: I fix the random seed
so each reported best-of-`k`-restarts is exactly reproducible, and I run every returned cap through
the verifier (and the independent triple scan at small `n`) before believing its size. The
multi-start changes which cap I report, not whether it is valid — validity is still guaranteed by
the admission rule — but I check anyway, because the whole point of this task is that no reported
size is ever taken on faith.
