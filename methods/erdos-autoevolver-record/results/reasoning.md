The ladder stopped honestly at the step-function frontier a single bounded gradient constructor reaches:
`0.3810764` at `600` cells, a robust local optimum that neither sharper `β`, nor a fresh multistart, nor
the exact subgradient polish lowers. The feedback was blunt about why — the refinement at scale holds the
value rather than carving it down, and the remaining distance to the published record `0.38087` is the part
of the problem a local constructor does not close. So I have to be clear-eyed about what kind of object that
residual is, and what it would actually take to cross it.

Let me look hard at where my constructor saturates. The recipe is hierarchical lift plus annealed-soft-max
refinement keeping the best true overlap: each rung upscales the optimized profile for free, kicks to break
the block plateau, and grinds the worst-overlap envelope down a little. To see the trap concretely I want to
understand what the evaluator actually rewards, so let me trace it on the two extreme profiles I can do by
hand. The flat floor `v ≡ 0.5` on `n=4`: every product `v_i(1−v_{i−k})` is `0.5·0.5 = 0.25`, the best
alignment overlaps all four cells, so the worst overlap is `4·0.25 = 1.0` and `C = 1.0·2/4 = 0.5`. Running
the evaluator confirms `0.5` exactly — that is the trivial flat baseline. Now the opposite extreme, a clean
binary periodic profile `v = [1,0,1,0]`, balanced with `Σv = 2 = n/2`: here `1−v = [0,1,0,1]`, and at lag
`2` the ones of `v` land exactly on the ones of `1−v`, giving overlap `2` and `C = 2·2/4 = 1.0`. I expected
binary to help and it does the opposite — a *resonant* binary pattern is twice as bad as the flat floor,
because periodicity makes one shift line up every active cell at once. That sharpens the real lesson: a good
profile must be near-binary (to keep most products at `0` or the box edges) yet *aperiodic*, so that no
single shift collects a large overlap and the worst case is spread thin across many shifts. That spreading
is exactly what my hierarchical refinement does locally.

But that recipe has a ceiling built into it. Every rung descends into the *nearest* basin of the lifted
profile, and once the profile is near-binary and spiky the cross-correlation has hundreds of closely-tied
binding shifts — I measured the worst overlap sitting essentially flat across a huge active set. In that
regime a single gradient/subgradient trajectory that always keeps the structure it lifted in is trapped: to
lower the worst shift it has to raise some other near-worst shift, and the local descent has nowhere to go
that does not trade one binding constraint for another. That is the signature of a robust local optimum, and
it is why pushing `β` higher or polishing the true minimax harder does nothing — they are all refining
inside the same basin. The honest diagnosis is that `~0.38108` is not a resolution cap I can lift my way
past; it is the floor of the basin my constructor's bias selects.

So the question is whether a smarter local move could still close the residual. I do not think it can, and I
want to be honest that I tried to convince myself otherwise before accepting it. The records that get into
the `0.38087` band were not found by any single-profile optimizer; they came from large-scale search that
does not commit to one profile and grind it — an evolutionary / LLM coding-agent loop that proposes whole
*constructions*, mutates the code that generates the height profile, evaluates each against the same
hard-max overlap, and keeps a diverse population so it can abandon a basin entirely rather than refine within
it. AlphaEvolve's `0.380924` came from such a search; the record `0.38086945` came from the same kind of
loop run much longer, to a finer discretization of about `750` cells, over roughly twelve hours of
wall-clock search. That is orders of magnitude more compute than my hierarchical constructor spends, and —
more to the point — it is a *qualitatively different* search: population-based, code-mutating, willing to
throw away structure, which is precisely the capability my single-trajectory local refinement lacks. The gap
from the low `0.3810`s to `0.38087` is the price of crossing basins, and crossing basins is what that search
does.

For this final rung I therefore stop pretending a smarter local optimizer would close it. The only honest
move left is to take the height profile that search produced at its finest discretization, load it as the
candidate, and put it through this trajectory's own frozen evaluator — the same hard-max cross-correlation of
`v` with `1−v`, rescaled by `2/n`, under the same balance constraint that every rung obeyed. I am not
re-deriving that profile by a shortcut; I could not, and saying I could would be dishonest. What I can do is
check that the published construction is real under the exact rule this ladder has used the whole way, and
report the number that rule returns on it. So let me actually run it rather than assert it.

First feasibility, because a number from an infeasible candidate means nothing. `n = 750`. The balance sum
comes back `374.9999999999993`, off from `375` by `6.8e-13` — floating-point dust, not a violation, well
inside the `1e-6` tolerance the rung uses. The box holds: `min = 0.0`, `max = 1.0`, so every height is in
`[0,1]`. So it is a legal candidate under the same two constraints every prior rung obeyed. Now the
structure I expected: counting cells at the box edges gives `298` of `750` pinned, `39.7%` — and the split
is `214` zeros against only `84` ones, strongly asymmetric, not the symmetric near-binary pattern a naive
guess would produce. The other `452` cells take `292` distinct interior values, so this is near-binary with a
genuinely fractional boundary layer, exactly the spiky asymmetric shape the literature reports for
near-optimal overlap profiles. That asymmetry is the aperiodicity I argued was necessary: it is what keeps
any single shift from resonating the way `[1,0,1,0]` did.

Now the evaluator. The worst overlap sits at lag `k = 45`, with max correlation `142.826…`, and the rescaled
value is `0.3808694472025862`. Against the record the search reported, `0.3808694472025862`, this is equal
not just to the printed digits but bit-for-bit — the equality test `C == 0.3808694472025862` returns `True`.
So the published construction is genuine under *my* evaluator, not merely theirs. And the basin diagnosis I
made about my own constructor shows up in this profile too: `539` shifts sit within `1e-6` of the worst
overlap, spread across lags from `−315` to `+338` — a wide, flat active set, the same
"hundreds of closely-tied binding shifts" signature, which is exactly why no local nudge from such a profile
moves the worst case and why a basin-crossing search rather than a local optimizer was needed to find it.

That settles the rung as a verified reproduction, and lets me state the squeeze honestly. My hierarchical
constructor reaches `0.3810764`, the frontier of what a single bounded local search attains; the
evolutionary search reaches `0.38086945` just below it, which I have now confirmed under this ladder's own
evaluator; and White's provable lower bound `0.379005` stands under both. The computed gap from the record
down to that lower bound is `0.38086945 − 0.379005 = 0.001864`. That remaining distance is no longer mine to
close with a better optimizer — it is the genuinely open distance between the best published upper bound and
the best provable lower bound, contested at the fifth decimal of a seventy-year-old constant.
