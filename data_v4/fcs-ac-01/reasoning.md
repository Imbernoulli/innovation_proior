The scale settles the shape of the answer before I write anything. `n = R*C` runs up to
`10^6`, and the number of board states is `n!` — even the `4 x 4` case has on the order of
`10^13` reachable states, and at a million cells `n!` is beyond any physical count. So I am
not allowed to touch the state graph at all; the intended solution has to be an `O(n)`
invariant. The other constraint that jumps out is `R, C >= 2`: never a 1-wide strip, always
a genuine two-dimensional grid. I file that away — a contract that goes out of its way to
forbid strips is telling me the clean theorem I am about to lean on only holds on true 2-D
boards. Input is `R`, `C`, then `n` integers for `A`, then `n` for `B`, both permutations of
`{0..n-1}` with `0` the blank; output is a single `YES`/`NO`. I keep `n` in a `long long`
for the `R*C` product and the `cell / C`, `cell % C` arithmetic, though at `10^6` an `int`
would already hold everything.

The literal reading — vertices are boards, edges are legal moves, is `B` in `A`'s connected
component — is a BFS I could write for a tiny board, but that component holds up to `n!/2`
states, so it is categorically infeasible, not merely slow. I have to decide reachability
without ever moving a tile: a property no move can change that, when matched between `A` and
`B`, *forces* reachability. An invariant, and a complete one.

So I look hard at a single move, because the invariant has to be built from what a move
leaves fixed. A move swaps the blank with a neighbour, and two things are true of every
move. Viewed as a permutation of cell-contents, a move is one transposition, hence an odd
permutation — it flips the parity of the arrangement. And the blank's grid position changes
by Manhattan distance exactly `1`, so the parity of how far the blank has travelled flips
too. The crucial point: both parities flip together, on every single move, without
exception. Define for the pair `(A, B)`:

- `parPi` = the parity of the permutation `pi` carrying `A`'s cell-contents to `B`'s (the
  blank `0` treated as just another label),
- `parMan` = the parity of the Manhattan distance between the blank cell in `A` and the one
  in `B`.

At `A == A`, `pi` is the identity (`parPi = 0`) and the blank hasn't moved (`parMan = 0`),
so `parPi XOR parMan = 0`. Each move toward `B`'s frame flips `parPi` (one transposition)
and flips `parMan` (blank steps once), leaving the XOR fixed. By induction
`parPi XOR parMan = 0` holds on every board reachable from `A`. So if `A` reaches `B`, then
`parPi == parMan`. That already kills the impossible cases: the 14-15 swap is a single
transposition of tiles `14` and `15` (`parPi = 1`) with the blank fixed in the corner
(`parMan = 0`), so `parPi != parMan` and the answer is `NO` — the notorious unsolvable
configuration falls straight out.

Necessity alone is not enough; I need `parPi == parMan` to *guarantee* reachability, or I
will print `YES` on unreachable boards. For a genuine 2-D board (`R, C >= 2`) this is the
classical reachability theorem for the `(n-1)`-puzzle: the reachable configurations are
*exactly* the parity-consistent ones — the puzzle group is the full alternating group `A_n`
once the blank's position is accounted for, one parity obstruction and nothing more. This is
exactly why the contract pins `R, C >= 2`: a `1 x C` strip is degenerate, the blank can only
shuffle along a line and can never reorder tiles, giving a completely different and much
finer invariant. Excluding strips keeps the clean parity theorem exact. So the decision is

> `reachable  <=>  same value multiset  AND  parPi == parMan`,

and everything left is mechanics computable in `O(n)`.

For `parPi` I avoid inversion counting — that is `O(n log n)` at best and fiddly. Parity has
a cleaner route through the cycle decomposition. Build `pi` over cells: `posB[v]` is the cell
where value `v` sits in `B`, and `pi[cell] = posB[A[cell]]` is where `A`'s content at `cell`
must land. The parity of a permutation on `n` elements is `(n - c) mod 2`, where `c` is the
number of cycles (a `k`-cycle is `k-1` transpositions; summing gives `n - c`). One linear
pass with a visited array counts the cycles.

Something about the order of `j = pi[j]` and `visited[j] = 1` looks off, so I trace the
smallest input that could expose it. Take the identity arrangement `A == B` on `n = 4`
cells, so `pi = [0,1,2,3]` — four fixed points, four cycles, `parPi` must be `(4-4)&1 = 0`.

Walk the inner loop at `i = 0`: I check `visited[0]` — false, so `cycles = 1`, set `j = 0`.
Loop condition `!visited[0]` is true: I execute `j = pi[0] = 0`, then `visited[0] = 1`. Back
to the condition `!visited[0]` — now false, exit. So far so good, one cell consumed. `i = 1`:
`visited[1]` false, `cycles = 2`, same dance, marks cell 1. Likewise `i = 2, 3`. Final
`cycles = 4`, `parPi = 0`. Correct here.

**Diagnosing the bug on a non-trivial cycle.** The identity hid it. Let me trace a real
cycle: `pi = [1, 0, 2, 3]` (a 2-cycle on `{0,1}` plus two fixed points), which is one
transposition, `n - c = 4 - 3 = 1`, so `parPi` must be `1`. At `i = 0`: `cycles = 1`,
`j = 0`. Condition `!visited[0]` true: `j = pi[0] = 1`, `visited[1] = 1`. Condition
`!visited[1]` — false! Loop exits having marked cell `1` but **never marked cell `0`**. Now
`i = 1`: `visited[1]` is true, skip. `i = 2, 3`: two more cycles. But `i = 0` was the start
of this cycle and got counted, while cell `0` itself is still unvisited — except the loop at
`i = 0` already moved past it. Worse, when I later reach... wait: cell `0` is never marked
visited at all. There is no `i` that revisits `0` because the outer loop only goes forward
and `i = 0` is done. So `0` stays unvisited forever, but the outer loop never returns to it.
The count came out `cycles = 1 (from i=0) + 1 (i=2) + 1 (i=3) = 3`, giving `parPi = (4-3)&1
= 1` — *accidentally* right here, but for the wrong reason: I got lucky that the unmarked
start cell happened not to spawn a phantom extra cycle. The real defect is the loop body
advances `j` **before** marking it, so the *starting* cell of each cycle is the one that
gets marked one step late, and on cycles of length `>= 2` the start cell `i` is left
unvisited. On some inputs that leaves a node that a later, smaller... no — `i` only
increases, so a left-behind start cell with index below the current `i` is simply lost,
never recounted; but a cycle whose minimum element is its *start* can get its bookkeeping
tangled when the traversal re-enters. I do not want to reason case-by-case about when the
luck holds. The fix is unambiguous: **mark `j` visited, then advance.**

Re-trace `pi = [1, 0, 2, 3]`. `i = 0`: `cycles = 1`, `j = 0`. `!visited[0]` true: mark `0`,
`j = pi[0] = 1`. `!visited[1]` true: mark `1`, `j = pi[1] = 0`. `!visited[0]` now false:
exit. Cells `0, 1` both marked — one cycle, correctly. `i = 1`: visited, skip. `i = 2`:
`cycles = 2`, mark `2`, `j = pi[2] = 2`, exit. `i = 3`: `cycles = 3`, mark `3`. Final
`cycles = 3`, `parPi = (4 - 3) & 1 = 1`. Correct, and now for the right reason: every cell
is marked exactly once, so `cycles` is exactly the number of orbits. This is the version I
keep.

The one place this transcription dies is the order of operations inside the cycle walk. It
has to be mark-then-advance:

```
int j = i;
while (!visited[j]) { visited[j] = 1; j = pi[j]; }
```

If I advance first — `{ j = pi[j]; visited[j] = 1; }` — the start cell of each cycle is
stepped over before it is ever marked, so on any cycle of length `>= 2` the entry cell stays
unvisited and the cycle count drifts. It is easy to miss because a lone 2-cycle like
`pi = [1,0,2,3]` still comes out right by luck. A 3-cycle exposes it: `pi = [1,2,0]` is a
single orbit (parity even, `n - c = 3 - 1 = 2`), but advancing-first marks only two of the
three cells, spread across two separate outer starts, reports `c = 2`, and flips the parity
to odd — a wrong answer. Marking first makes every cell get marked exactly once, so `cycles`
is exactly the orbit count and `parPi = (n - cycles) & 1`.

`parMan` is two lookups: the blank cell `za` in `A` and `zb` in `B`, converted to `(row,
col)` via `/C` and `%C`, then `|ra-rb| + |ca-cb|` read for its low bit. One guard goes in
front of everything: the contract promises both boards are permutations of the same set, but
a malformed pair (a repeated or out-of-range value) must print `NO` rather than crash — and
such a value would send `posB[B[cell]] = cell` out of bounds. So I first check each board is
a permutation of `{0..n-1}` with a `seen` array; either failing (which also covers a genuine
multiset mismatch, itself trivially unreachable) prints `NO` and stops.

I cannot eyeball a million-cell board, so I trust the `O(n)` decision only against the BFS I
rejected for speed but which is exactly correct on tiny boards. Over hundreds of random 2-D
boards up to `3 x 3` plus the hand cases — identical boards (`YES`), a single slide (both
parities flip to `1`, `YES`), a single tile transposition with the blank fixed (the 14-15
family, `NO`), differing multisets (`NO`, no crash) — the two agree on every pair, and the
`4 x 4` 14-15 swap returns `NO`. At `n = 10^6` the work is two linear passes plus a cycle
walk touching each cell once, `int` cells with `long long` for `n` and the Manhattan terms,
about `20 MB`, and `sync_with_stdio(false)` keeps the read of two million integers fast. The
full self-contained module is in the answer.
