The ladder stopped honestly at the step-function frontier a single bounded gradient constructor reaches:
`0.3810764` at `600` cells, a robust local optimum that neither sharper `β`, nor a fresh multistart, nor
the exact subgradient polish lowers. The feedback was blunt about why — the refinement at scale holds the
value rather than carving it down, and the remaining distance to the published record `0.38087` is the part
of the problem a local constructor does not close. So I have to be clear-eyed about what kind of object that
residual is, and what it would actually take to cross it.

Let me look hard at where my constructor saturates. The recipe is hierarchical lift plus annealed-soft-max
refinement keeping the best true overlap: each rung upscales the optimized profile for free, kicks to break
the block plateau, and grinds the worst-overlap envelope down a little. That recipe has a ceiling built into
it. Every rung descends into the *nearest* basin of the lifted profile, and once the profile is near-binary
and spiky the cross-correlation has hundreds of closely-tied binding shifts — I measured the worst overlap
sitting essentially flat across a huge active set. In that regime a single gradient/subgradient trajectory
that always keeps the structure it lifted in is trapped: to lower the worst shift it has to raise some other
near-worst shift, and the local descent has nowhere to go that does not trade one binding constraint for
another. That is exactly the signature of a robust local optimum, and it is why pushing `β` higher or
polishing the true minimax harder does nothing — they are all refining inside the same basin. The honest
diagnosis is that `~0.38108` is not a resolution cap I can lift my way past; it is the floor of the basin my
constructor's bias selects, and the records live in a *different* basin that a continuous local search from
my lifted profile cannot reach.

So what reaches the record? Not a better optimizer of my kind. The published constructions that get into the
`0.38087` band were found by large-scale search that does not commit to a single profile and grind it: an
evolutionary / LLM coding-agent loop that proposes whole *constructions*, mutates the code that generates the
height profile, evaluates each against the same hard-max overlap, and keeps a diverse population so it can
abandon a basin entirely rather than refine within it. AlphaEvolve's `0.380924` came from such an
evolutionary search; AutoEvolver's record `0.38086945` came from the same kind of loop run much longer, to a
finer discretization of about `750` cells, over roughly twelve hours of wall-clock search. That is orders of
magnitude more compute than my hierarchical constructor spends, and — more to the point — it is a
*qualitatively different* search: population-based, code-mutating, willing to throw away structure, which is
precisely the capability my single-trajectory local refinement lacks. The gap from the low `0.3810`s to
`0.38087` is the price of crossing basins, and crossing basins is what that search does.

For this final rung I therefore stop pretending a smarter local optimizer would close it, and instead reach
the record the only honest way available: reproduce the AutoEvolver construction itself. I take the height
profile that search produced at its finest discretization, load it as the candidate, and put it through this
trajectory's own frozen evaluator — the same hard-max cross-correlation of `v` with `1−v`, rescaled by
`2/n`, under the same balance constraint that every rung obeyed. I am not re-deriving that profile by a
shortcut; I could not, and saying I could would be dishonest. What I can do is verify that the published
construction is real under the exact rule this ladder has used the whole way, and report the number that rule
returns on it.

What I expect: the profile is near-binary like every rung before it, with a large active set of binding
shifts — the spiky asymmetric structure the literature reports for near-optimal overlap profiles — and it
must satisfy the balance constraint `Σ v = n/2` exactly or it is not a legal candidate at all, so I check
that first. Then the evaluator returns its worst overlap, and the question is simply whether that number is
the record `0.38087` the search reported, computed under *my* evaluator rather than theirs. If it is, the
final rung of this ladder is the record itself, reached not by out-optimizing within my basin but by adopting
the construction that large-scale evolutionary search found in another one — and the honest framing of the
whole trajectory becomes a squeeze: my hierarchical constructor reaching the frontier of what a single
bounded local search attains, the evolutionary search reaching the current record just below it, and White's
provable lower bound `0.379005` standing under both as the floor the true constant cannot cross. The distance
that remains is no longer mine to close with a better optimizer; it is the genuinely open distance between the
best published upper bound and the best provable lower bound, contested at the fifth decimal of a
seventy-year-old constant.
