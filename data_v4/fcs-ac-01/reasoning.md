**Reading the problem and pinning the contract.** I have two boards `A` and `B`, each an
`R x C` grid holding the values `0..R*C-1` where `0` is the blank. A move slides a tile
into the blank, i.e. the blank swaps with an orthogonal neighbour. I must decide whether
some sequence of moves carries `A` to `B`, printing `YES` or `NO`. Constraints first,
because they decide everything: `R, C >= 2` and `n = R*C <= 10^6`. So the board is a
genuine 2-D grid (no 1-wide strips), and `n` is up to a million. I read `R` then `C`, then
`n` integers for `A`, then `n` integers for `B`. `R*C` can be `10^6`, which fits in 32
bits, but I will hold `n` in a `long long` anyway so that `R*C` never risks a surprise and
so cell arithmetic like `cell / C` stays comfortable. Output is one token.

**The obvious approach, stated honestly.** The literal reading of "can `A` reach `B`?" is a
graph reachability question: vertices are board states, edges are legal moves, and I want
to know whether `B` lies in the connected component of `A`. The natural tool is BFS from
`A` until I either dequeue `B` or exhaust the component. This is unambiguously correct, and
for a tiny board I would just write it. So let me size it before trusting it.

**Showing the obvious approach cannot survive the constraints.** The number of states is
`n!`. For the `4 x 4` board that is `16! ~ 2.1 * 10^13`; the reachable component is half of
that, `~10^13` states. BFS would need to enumerate and store on the order of `10^13`
boards, each of size `16`. That is already impossible in 2 seconds and 256 MB, and the
problem allows `n` up to `10^6`, where `n!` is astronomically beyond any physical count.
Search over states is not slow — it is categorically infeasible. I have to decide
reachability *without ever moving a tile*. That means I need a property of a board that (a)
no move can change, so equal boards-modulo-reachability share it, and (b) is strong enough
that sharing it forces reachability. An invariant, and a complete one.

**Hunting for what a move actually preserves.** Let me look hard at a single move, because
the invariant has to be built from whatever a move leaves fixed. A move swaps the blank
with a neighbour. Two things are true of every move, and I want both:

1. *As a permutation, a move is one transposition.* If I think of the board as a function
   from cells to values, a move exchanges the contents of exactly two cells. A single
   transposition is an **odd** permutation, so it flips the parity of the arrangement.
2. *The blank moves by exactly one grid step.* Its Manhattan position changes by `1` each
   move, so the parity of "how far the blank has travelled" also flips on every move.

Here is the crucial observation: **both parities flip together, on every single move,
without exception.** Whatever a move does to the permutation parity, it does the identical
thing to the blank-displacement parity. So their relationship is frozen. Concretely, define
for the pair `(A, B)`:

- `parPi` = the parity of the permutation `pi` that carries the contents of `A` to the
  contents of `B` (treating the blank `0` as just another label),
- `parMan` = the parity of the Manhattan distance between the blank cell in `A` and the
  blank cell in `B`.

Start at `A == A`: `pi` is the identity (even, `parPi = 0`) and the blank hasn't moved
(`parMan = 0`), so `parPi XOR parMan = 0`. Now apply any move to go from the current board
toward `B`'s frame: it flips `parPi` (one transposition) and flips `parMan` (blank steps
once). The XOR is unchanged. By induction, `parPi XOR parMan = 0` holds for *every* board
reachable from `A`. Therefore:

> If `A` reaches `B`, then `parPi == parMan`.

That is necessity, and it already kills the impossible cases. The 14-15 swap is exactly
this: `pi` is a single transposition of tiles `14` and `15` (odd, `parPi = 1`) while the
blank sits in the corner in both boards (`parMan = 0`), so `parPi != parMan` and the answer
is `NO` — the classic unsolvable configuration falls straight out.

**Is the invariant also sufficient?** Necessity alone is not enough; I need `parPi == parMan`
to *guarantee* reachability, or I will print `YES` on unreachable boards. For a genuine 2-D
board (`R, C >= 2`) this is the classical reachability theorem for the `(n-1)`-puzzle
(Wilson's theorem on the puzzle group): on any grid with both dimensions at least `2`, the
reachable configurations are *exactly* the even-parity-relative-to-blank-distance ones —
the puzzle group is the full alternating group `A_n` once you account for the blank's
position. There is exactly one parity obstruction and nothing more. So on a 2-D board,
matching the single invariant is necessary *and* sufficient. (This is precisely why the
contract pins `R, C >= 2`: a `1 x C` strip is degenerate — the blank can only shuffle along
a line and can never reorder tiles, giving a completely different, much finer invariant. By
excluding strips I keep the clean parity theorem exact.) So the decision is:

> `reachable  <=>  same value multiset  AND  parPi == parMan`.

This is the entire problem. Everything now is mechanics: compute `parPi` and `parMan` in
`O(n)`.

**Computing the permutation parity in linear time.** I do not want inversion counting
(that is `O(n log n)` at best and fiddly); permutation parity has a cleaner route. Build
`pi` as a permutation of cells: `posB[v]` is the cell where value `v` sits in `B`, and
`pi[cell] = posB[A[cell]]` is where `A`'s content at `cell` needs to go. The parity of a
permutation on `n` elements is `(n - c) mod 2`, where `c` is the number of cycles in its
cycle decomposition (each `k`-cycle is `k-1` transpositions; summing, `n - c`). One linear
pass with a visited array counts the cycles. The blank parities are just two cell lookups
and a Manhattan distance. All `O(n)`.

**First implementation — and I will trace it, because clean math transcribes dirty.** My
first cut of the core:

```
vector<int> posB(n);
for (int cell = 0; cell < n; cell++) posB[B[cell]] = cell;
vector<int> pi(n);
for (int cell = 0; cell < n; cell++) pi[cell] = posB[A[cell]];

int cycles = 0;
vector<char> visited(n, 0);
for (int i = 0; i < n; i++) {
    if (!visited[i]) {
        cycles++;
        int j = i;
        while (!visited[j]) { j = pi[j]; visited[j] = 1; }   // <-- suspicious
    }
}
int parPi = (n - cycles) & 1;
```

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

**Fixing and re-verifying the cycle counter.** Correct body — mark first, move second, and
test the condition on the cell I am about to mark:

```
int j = i;
while (!visited[j]) { visited[j] = 1; j = pi[j]; }
```

Re-trace `pi = [1, 0, 2, 3]`. `i = 0`: `cycles = 1`, `j = 0`. `!visited[0]` true: mark `0`,
`j = pi[0] = 1`. `!visited[1]` true: mark `1`, `j = pi[1] = 0`. `!visited[0]` now false:
exit. Cells `0, 1` both marked — one cycle, correctly. `i = 1`: visited, skip. `i = 2`:
`cycles = 2`, mark `2`, `j = pi[2] = 2`, exit. `i = 3`: `cycles = 3`, mark `3`. Final
`cycles = 3`, `parPi = (4 - 3) & 1 = 1`. Correct, and now for the right reason: every cell
is marked exactly once, so `cycles` is exactly the number of orbits. This is the version I
keep.

**Wiring in the blank parity and the multiset guard.** `parMan`: find the blank cell `za`
in `A` and `zb` in `B`, convert to `(row, col)` via `/C` and `%C`, take
`|ra-rb| + |ca-cb|` and read its low bit. The decision is `parPi == parMan ? YES : NO`.
One more guard: the contract promises `A` and `B` are permutations of the same set, but a
malformed pair (a repeated or out-of-range value) must print `NO`, never crash — and if the
multisets differ, the boards are trivially unreachable. So before any parity work I verify
each board is a permutation of `{0..n-1}` with a `seen` array; if either fails I print `NO`
and stop. This also protects `posB[B[cell]] = cell` from an out-of-range index.

**Edge cases, deliberately, because this is where this kind of code dies.**
- *Identical boards* (`A == B`): `pi` is the identity, `parPi = 0`; blank doesn't move,
  `parMan = 0`; equal -> `YES`. Correct, reflexivity holds.
- *Single move* (blank slides one cell): `pi` is one transposition (`parPi = 1`) and the
  blank moves Manhattan `1` (`parMan = 1`); equal -> `YES`. Correct.
- *Single tile transposition, blank fixed* (e.g. swap two non-blank tiles, leave blank put):
  `parPi = 1`, `parMan = 0` -> `NO`. This is the 14-15 swap family; correct.
- *Smallest board* `2 x 2`: only `n!/2 = 12` reachable states; I cross-checked the formula
  against full BFS on `2 x 2`, `2 x 3`, `3 x 2`, `2 x 4`, `4 x 2`, `3 x 3` and it matches on
  every pair.
- *Multiset mismatch / malformed*: caught by the permutation guard -> `NO`, no crash.
- *Scale* `n = 10^6`: two linear passes plus a cycle walk that touches each cell once;
  `int` cells, `long long` for `n` and the Manhattan terms; `~20 MB`, well under limits.
  `sync_with_stdio(false)` keeps the read of two million integers fast.

**Verifying against a slow oracle.** I cannot eyeball a million-cell board, so I trust this
only after a differential test: a BFS brute force (the very state search I rejected for
performance, but perfectly correct on tiny boards) versus this `O(n)` decision over many
hundreds of random small `2-D` boards plus the hand-built edge cases above. They agree on
every case; the `4 x 4` 14-15 swap returns `NO` and the solved board returns `YES`. The
invariant I derived is the thing the brute force confirms, which is the evidence I trust.

**Final solution.** I convinced myself the *idea* is right by reducing every move to a
simultaneous flip of two parities and invoking the puzzle-group theorem for the sufficiency
direction, and I convinced myself the *code* is right by tracing the cycle counter to a
precise mark-then-advance bug, fixing it, and differential-testing against BFS. That is what
I ship — one self-contained file, the `O(n)` parity decision I can defend rather than the
state search I proved infeasible:

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
    Parity-Invariant Reachability (generalized R x C sliding-tile puzzle).

    A board of R rows and C columns (R >= 2, C >= 2) holds the values
    0..R*C-1, where 0 is the blank. A move slides a tile orthogonally
    adjacent to the blank into the blank cell (the blank swaps with a
    4-neighbour). Given boards A and B (permutations of {0..R*C-1}), decide
    whether A can be turned into B. Print YES or NO.

    Insight (the whole problem): every move is ONE transposition of the
    cell-permutation (so it flips permutation parity) AND it moves the blank
    by Manhattan distance exactly 1 (so it flips the parity of the blank's
    Manhattan displacement). Hence the quantity

        parity(pi)  XOR  parity(Manhattan(blank_A, blank_B))

    where pi is the permutation carrying A's cell-contents to B's, is
    invariant under every move and equals 0 in the start state (A == A).
    For a genuine 2D board (R,C >= 2) this single invariant is also
    sufficient (classic 15-puzzle / Wilson's theorem). So:

        reachable  <=>  same multiset  AND  parity(pi) == parity(Manhattan).

    We compute parity(pi) in O(n) via cycle decomposition and read the blank
    positions directly. O(n) time, O(n) memory; n = R*C up to 1e6.
*/

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long R, C;
    if (!(cin >> R >> C)) return 0;
    long long n = R * C;

    vector<int> A(n), B(n);
    for (auto &x : A) cin >> x;
    for (auto &x : B) cin >> x;

    // Precondition: A and B must be permutations of the same multiset.
    // The contract guarantees both are permutations of {0..n-1}; we still
    // verify defensively so a malformed pair is reported NO, never crashes.
    auto valid_perm = [&](const vector<int> &V) -> bool {
        vector<char> seen(n, 0);
        for (int v : V) {
            if (v < 0 || v >= n || seen[v]) return false;
            seen[v] = 1;
        }
        return true;
    };
    if (!valid_perm(A) || !valid_perm(B)) {
        cout << "NO\n";
        return 0;
    }

    // posB[v] = cell index of value v in board B.
    vector<int> posB(n);
    for (int cell = 0; cell < n; cell++) posB[B[cell]] = cell;

    // pi[cell] = the cell that A[cell]'s value occupies in B.
    // parity(pi) = (n - number_of_cycles) mod 2.
    vector<int> pi(n);
    for (int cell = 0; cell < n; cell++) pi[cell] = posB[A[cell]];

    vector<char> visited(n, 0);
    long long cycles = 0;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            cycles++;
            int j = i;
            while (!visited[j]) {
                visited[j] = 1;
                j = pi[j];
            }
        }
    }
    int parPi = (int)((n - cycles) & 1LL);

    // Manhattan distance between the two blank cells, parity only.
    int za = 0, zb = 0;
    for (int cell = 0; cell < n; cell++) {
        if (A[cell] == 0) za = cell;
        if (B[cell] == 0) zb = cell;
    }
    long long ra = za / C, ca = za % C;
    long long rb = zb / C, cb = zb % C;
    int parMan = (int)((llabs(ra - rb) + llabs(ca - cb)) & 1LL);

    cout << (parPi == parMan ? "YES" : "NO") << "\n";
    return 0;
}
```

**Causal recap.** Direct state search is correct but enumerates up to `n!` boards, so it
cannot run; I needed a complete invariant instead. Every move is simultaneously one
transposition (flipping permutation parity) and one blank step (flipping blank-Manhattan
parity), so `parity(pi) XOR parity(Manhattan)` is preserved from its starting value `0` —
that is necessity, and it is exactly what makes the 14-15 swap unreachable. For a true 2-D
board the puzzle-group theorem makes the same invariant sufficient, so reachability reduces
to `same multiset AND parPi == parMan`. I compute `parPi` as `(n - cycles) mod 2` from a
linear cycle walk — whose first version advanced the pointer before marking and left each
cycle's start cell unvisited, caught by tracing `pi = [1,0,2,3]`; marking before advancing
fixes it — and `parMan` from two blank-cell lookups, the whole decision running in `O(n)`
and matching BFS on every small board I tested.
