# Reasoning: Reconfiguration Routing on a grid

The score here is `1000 * LB / max(L, LB)`, where `L` is how many single-token
steps I emit and `LB` is the sum over tokens of each one's own shortest-path
distance to its target with nobody else on the board. Two features of that
formula decide the whole approach. First, *any* illegal move — stepping into a
wall, off the grid, or onto a cell another token occupies — or any token left off
its target floors the entire seed to 0; there is no partial credit for a
nearly-legal plan. Second, `LB` pretends each token has the board to itself, so
the score I lose is exactly the moves I spend untangling tokens from one
another's way. The grids are small (12–30 on a side), up to 40 tokens, and the
generator guarantees at least three blank cells per token — a roomy board where
tokens can genuinely slide past each other rather than gridlock. So the game is:
never be infeasible, and keep the total move count as close as possible to the
sum of independent shortest paths.

Because the floor is so punishing, my first commitment is a plan that is *always*
legal, even a wasteful one; only once I have that in hand do I chase the move
count. A clever plan that is occasionally infeasible scores zero on those seeds
and drags the mean below a dumb-but-legal plan.

The simplest always-legal thing is to abandon parallelism and move tokens one at
a time: route token 0 fully to its target, then token 1, and so on; while one
token moves, every other sits as an obstacle. I BFS the active token toward its
target over free cells, walk the path when it is clear, and when a stationary
token blocks the path I shove it out of the way through the blank space.

That shove is where the sliding-puzzle nature bites, and my first attempt was
wrong. My instinct was to BFS for a blank and slide tokens along that path — but a blank's
BFS happily routes through *other* blank cells, and when the next cell on its
path is itself empty there is no token there to slide; I indexed a phantom token
and the address sanitizer caught a heap overflow inside the slide loop. The fix
reshaped the primitive: instead of walking a blank along an arbitrary free path,
I BFS *from the cell I want vacated* over token-occupied cells until I reach the
nearest blank, which gives a chain blank—token—…—target, and I slide each token
one step toward the target so the blank bubbles up into the cell I wanted
emptied. Every step of that chain has a real token to move, so the phantom-index
bug cannot recur, and on a connected board with at least one blank this can
always vacate the next cell — the sequential mover always finishes.

It is legal but wasteful exactly where the scoring punishes hardest: a token
finished early still gets shoved aside for later tokens, later tokens detour
around every earlier obstacle, and the blank-bubbling itself burns moves. Each is
a move above `LB`. That gap is what a real method has to close.

The tempting way to close it is to let every token follow its own shortest path
at once — that would hit `LB` exactly, and it is almost always illegal, because
two shortest paths want the same cell at the same instant, or two tokens try to
swap across an edge. I cannot overlay independent paths; the tokens have to
coordinate, moving in parallel but yielding to each other while each stays near
its own shortest path. That is multi-agent path finding, and the standard tool is
prioritized planning with space-time A\*.

The idea is to give the tokens a priority order and plan them one at a time, but
not *route* them one at a time — each token reserves the cells it occupies *at
each time step*, and every lower-priority token must plan a path avoiding those
reserved `(cell, time)` slots. So the search is over **space-time**: a state is
`(cell, t)`, the moves are "step to a free neighbour" or "wait in place"
(advancing `t` by one), and any `(cell, t)` a higher-priority token occupies is
forbidden. I run A\* in this expanded graph with the admissible heuristic being
the single-token BFS distance from the cell to the goal, ignoring everyone else.
Because each token sees the others only as time-stamped obstacles, it can weave
through them — waiting a step to let someone pass, taking a short detour — while
still landing near its own shortest path. The reservations live in a hash set
keyed by `(cell, t)` packed into a 64-bit integer, so each collision test is
O(1); that is what keeps the search fast enough to plan dozens of tokens inside
the two-second budget. Sorting by difficulty (largest single-token distance
first) lets the most constrained token claim space-time before the easy ones box
it in.

Several details have to be exactly right, and I worked them out by running into
each one.

Once a token reaches its target it stays there forever, so I "park" its goal:
reserve that cell for *all* later times, which stops every lower-priority token
from ever stepping onto an already-placed token. The necessary complement is that
a token's *start* cell is a static obstacle for everyone planned before it moves
— otherwise a higher-priority token could plan straight through where a
not-yet-planned token still sits.

Serialization is the next trap, and it gave me a real bug. The output is not a
set of parallel timestep-snapshots; it is a flat stream of single-token actions,
so the time-expanded plan has to serialize into one-token-one-step moves that
are *themselves* collision-free in whatever order I emit them. My first
serialization just walked each timestep and emitted every token that moved — and
on several seeds the scorer reported a collision at the very first action, two
tokens landing on one cell. The cause is that within a single timestep one token
can move *into* a cell another token is moving *out of* (a "follow"), or several
tokens can rotate around a cycle; emitting those one at a time is not
collision-free. I fixed it at the planning level, not the output level: I forbid
a token from entering a cell that is **occupied at the start of the timestep**.
With that no-chaining rule, every move in a timestep lands on a cell that was
empty before the timestep began, so the moves are pairwise independent and I can
serialize them in any order. To make the rule hold across priorities I also add
an *approach reservation*: when a token enters cell `X` at time `t+1`, I reserve
`X` at time `t` too, so no later token will be sitting on `X` at time `t`. After
those two changes the serialized plans come out clean.

Even so, a batch of seeds still produced a collision deep in the plan.
Instrumenting the parallel plan turned up pairs like "token 4 enters cell 235 at
t=18, but token 6 is parked on 235." Token 6 had a *lower* priority and its goal
happened to be 235; it reached that goal early and parked, but token 4 (higher
priority, planned earlier) had a path crossing 235 at t=18 and never knew 235
would become someone's permanent home. The fix is that a token may only *finish*
at its goal once the goal is free for all later times under the current
reservations: I compute, per token, the latest time its goal cell is reserved by
anyone higher-priority, and refuse to accept the goal as terminal in A\* until
after that time — the token waits, if it must, before settling. That single guard
removed the last class of collisions.

Prioritized planning is not complete: some solvable instances — head-on swaps,
cyclic dependencies where A's goal is B's start and vice versa — have no priority
order in which everyone's A\* succeeds. So when a token's space-time A\* returns
no path, I do the standard windowed re-plan: bump the offending token to the
front of the order so it claims space-time first, and retry; every few attempts I
take a full random shuffle of the order to escape cyclic priority dependencies
the deterministic bump cannot break. I cap this at a slice of the time budget so
I always leave time for the fallback. On most seeds the very first difficulty
order succeeds immediately; a handful need a few re-prioritizations; and the
genuine swap instances that never succeed under any order fall through to the
sequential mover.

So the final solver runs both ingredients and emits whichever feasible plan has
fewer actions: prioritized planning (with re-prioritization) for a near-`LB`
parallel plan, and the sequential placement on a few orders as a safety net that
guarantees feasibility on the swap instances prioritized planning cannot crack
and occasionally wins on tiny instances. For the deadlock cases where the
fallback would just equal the baseline, I also try short greedy
distance-reducing prefixes before handing off to the sequential finisher, keeping
any combination that comes out shorter — this never risks feasibility, since a
candidate is used only after both a replayable prefix and a successful sequential
suffix have been built. The baseline I measure against is the *pure* sequential
one-at-a-time mover: the same robust sliding-puzzle placement in plain input
order, taking the first feasible plan with no parallelism and no order search.

One more robustness lesson came out of the sequential placement. When I "freeze"
a placed token (treat its cell as a wall so it is never disturbed again), the
freeze can carve a new dead-end and cut a later token off from its goal — even
though the board's 1-wide dead-ends were eroded away to begin with. So I make the
placement connectivity-aware: before committing to freeze a goal cell, I check
that the remaining free region (minus that cell) is still connected and still
reaches every unplaced token's goal with at least one blank, and I prefer to
place a token whose freezing keeps connectivity, freezing a "risky" one only when
nothing safer is available. The comfortable blank margin — at least three blanks
per token — keeps the region roomy enough for tokens to slide past one another,
which is the regime where coordinated space-time planning, not brute parking,
earns its score.

Running the solver against the local scorer on the frozen seed set, the
discipline I hold to is that *every* output must be feasible (the scorer confirms
all tokens on target) and the mean must strictly beat the pure sequential
baseline's. On the twenty-seed set every output is feasible, the mean score is
around 933 against the baseline's ~900, and the solver is never worse than the
baseline on any single seed — it wins on most and ties on a few small instances
where the parallel and sequential plans coincide. Widening to forty seeds keeps
every output feasible with the slowest run inside the time budget.
