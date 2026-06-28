**Problem.** A generalized sliding-tile puzzle on an `R x C` grid (`R, C >= 2`,
`R*C <= 10^6`) holds the values `0..R*C-1`, where `0` is the blank. A move slides a tile
into the blank, i.e. the blank swaps with an orthogonal neighbour. Given a start board `A`
and a target board `B` (both permutations of `{0..R*C-1}`, given row-major on stdin),
decide whether `A` can be transformed into `B`. Print `YES` or `NO`.

**Why the obvious approach is hopeless.** Treating this as graph reachability and running
BFS from `A` is correct but enumerates the component of `A`, which has up to `(R*C)!/2`
states — over `10^13` already for the `4 x 4` board, and unbounded at the stated scale.
Search cannot run; reachability must be decided without moving a single tile.

**Key idea — a complete parity invariant.** Every move does two things *simultaneously and
always*: it is one transposition of the cell-permutation (flipping its parity), and it moves
the blank by Manhattan distance exactly `1` (flipping the parity of the blank's
displacement). So if `pi` is the permutation carrying `A`'s cell-contents to `B`'s, the
quantity

```
parity(pi)  XOR  parity(Manhattan(blank_A, blank_B))
```

is preserved by every move. It is `0` at the start (`A == A`: identity permutation, blank
unmoved), so it stays `0` along any reachable path — that is necessity, and it is exactly
why the famous 14-15 swap (a single tile transposition with the blank fixed) is
unreachable. For a genuine 2-D board (`R, C >= 2`) the classical `(n-1)`-puzzle / Wilson's
group theorem makes this single invariant *sufficient* as well: the reachable set is
precisely the parity-consistent one. Hence

```
reachable  <=>  same multiset  AND  parity(pi) == parity(Manhattan).
```

**Pitfalls.**
- *Cycle-walk ordering.* Permutation parity is `(n - cycles) mod 2`. The natural cycle loop
  must mark a cell visited **before** advancing the pointer; advancing first leaves each
  cycle's start cell unvisited and miscounts cycles. Trace `pi = [1,0,2,3]` (one
  transposition, parity `1`) to expose it.
- *Don't reach for inversion counting.* Counting inversions is `O(n log n)` and easy to get
  subtly wrong; cycle decomposition gives the parity directly in `O(n)`.
- *Strips are excluded for a reason.* On a `1 x C` board the blank can never reorder tiles,
  so the parity invariant is *not* sufficient there. The contract pins `R, C >= 2`, where the
  theorem is exact.
- *Index safety.* Use `posB[B[cell]] = cell`; a malformed board (out-of-range or repeated
  value) would index out of bounds, so guard with a permutation check first.

**Edge cases.** Identical boards -> `YES` (identity, both parities `0`). Single move ->
`YES` (both parities flip to `1`). Single tile transposition with blank fixed -> `NO`
(`parPi=1`, `parMan=0`). The `4 x 4` 14-15 swap -> `NO`. Differing multisets or malformed
input -> `NO`, never a crash. Smallest board `2 x 2` and other tiny boards cross-checked
against full BFS.

**Complexity.** Two linear passes plus one cycle walk that touches each cell once: `O(n)`
time, `O(n)` memory (`n = R*C`). Comfortably within 2 s / 256 MB at `n = 10^6`.

**Code.**

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
