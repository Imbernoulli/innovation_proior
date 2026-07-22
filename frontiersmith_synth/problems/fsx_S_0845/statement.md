# The Bloc-Proof Ballot

A jurisdiction runs a single-winner ranked election: **100 voters**, **10 candidates**
(indices `0..9`). Every voter casts a full strict preference order over all 10 candidates.

Your program **is the aggregation rule**. It is invoked as a standalone process, once per
election it is shown, and must read one election from stdin and print its winner to stdout:

```
stdin  (JSON): {"num_voters": 100, "num_candidates": 10,
                "ballots": [[c_0, c_1, ..., c_9], ...]}   # 100 permutations of 0..9,
                                                            # most-preferred candidate first
stdout (JSON): {"winner": w}                               # 0 <= w < 10, integer
```

There are **10 fixed base elections**. For each one, the evaluator additionally maintains a
designated **bloc size b** (5, 10, or 20, varying by election) and a **published, deterministic
manipulation sweep**: three ways a coordinated bloc of exactly `b` voters can rewrite *only their
own* ballots (the other `100 - b` stay exactly as given). Your rule is invoked on the untouched
election AND on all three rewrites of it — four calls total per election, each seeing only the
ballots for that one call. The three sweep recipes (fixed, not tuned to your program):

1. **COMPROMISE.** `b` voters who sincerely favor the current frontrunner instead all rank a
   different, weaker contender first, trying to push that contender past the frontrunner.
2. **BURY.** `b` voters who sincerely rank the strongest candidate near the top instead rank that
   candidate dead last, keeping everyone else's relative order.
3. **CYCLE-INJECT.** `b` voters, split into three groups as equal as possible (sizes differ by at
   most one), each group unanimously imposes one of the three cyclic rotations of the current
   top-3 candidates (group 1: `X,Y,Z`; group 2: `Y,Z,X`; group 3: `Z,X,Y`), trying to manufacture
   a majority cycle among them.

**Scoring.** Every voter has a hidden true utility for every candidate (never shown to your
program). A candidate's *social cost* is the sum over voters of `(1 - utility)`; the true-optimal
candidate `c*` minimizes it. For election `i`, let `D_i` be the **worst** (largest) ratio
`cost(your winner) / cost(c*)` over the four calls (untouched + the three rewrites) — this is the
election's *distortion under the sweep*, and you want it as close to 1 as possible. The evaluator
also computes the same worst-case-over-the-sweep distortion for a plain plurality rule internally,
`D_i(plurality)`, and normalizes:

```
r_i = clamp( 0.1 * D_i(plurality) / D_i(your rule), 0, 1 )
score = mean(r_i over the 10 elections)
```

Reproducing plain plurality scores `r_i = 0.1` on every election — that is the "no defense"
anchor. A rule whose winner barely moves under the sweep scores well above 0.1; a rule that is
*more* exploitable than plurality scores below 0.1.

**Feasibility.** `winner` must be an integer in `[0, 10)`. Any malformed output, wrong type,
out-of-range value, crash, timeout, or non-JSON on *any* of the four calls for an election scores
that election `0.0` — a rule that cannot even produce a valid winner under a bloc-rewritten input
has failed the robustness requirement by construction.

**Notes for design.** 5 of the 10 elections have a clear, well-supported frontrunner but also a
smaller, distinctly lower-quality runner-up faction — exactly the setup the COMPROMISE recipe is
built to exploit. 2 of the 10 elections instead start from (or the CYCLE-INJECT recipe can create)
a genuine 3-way majority cycle among the top candidates, so a rule that only ever handles a clean
Condorcet winner is also tested. The remaining elections keep comfortable margins, where the sweep
cannot flip anything, to check you aren't sacrificing accuracy for paranoia. Nothing here requires
literal strategy-proofness (impossible for a non-trivial rule) — the objective is a rule whose
*winner function has low sensitivity* to a `b`-ballot rewrite of these specific sweep profiles.

Time limit: 15s per call (a generous safety net; a well-designed rule finishes in well under a
second on 100 voters x 10 candidates). Memory: 512MB.
