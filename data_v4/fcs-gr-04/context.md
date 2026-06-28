# Reconstruct a string from the multiset of its k-mers

## Research question

A long string was broken into all of its overlapping substrings of length `k` (its **k-mers**),
the pieces were shuffled, and the original ordering was thrown away. You are handed `k` and the
multiset of `m` k-mers (each k-mer appears in the list with the multiplicity it had as a window of
the original string). Reconstruct **one** string whose multiset of length-`k` windows is exactly the
given list — equivalently, a string that uses every supplied k-mer exactly once as a consecutive
substring. If no such string exists, report that the input is unreconstructable.

This is the model problem behind shotgun sequence assembly: you never see the whole string, only a
bag of short, overlapping fragments, and you must stitch them back into something consistent with all
of them at once. The catch is that "stitch the fragments" sounds like a search or a dynamic program
over orderings, and at the stated scale that framing does not survive.

## Input / output contract

- Input (stdin):
  - the first line has two integers `k` and `m` (`2 <= k <= 30`, `0 <= m <= 2*10^5`);
  - then `m` k-mers follow, whitespace-separated, each a string of exactly `k` lowercase letters
    (`a`–`z`). The same k-mer may appear multiple times; multiplicity is significant.
- Output (stdout):
  - if a reconstruction exists, print **one** valid superstring on a single line. It will have
    length `(k-1) + m`. Any valid reconstruction is accepted (the answer need not be unique).
  - otherwise print `IMPOSSIBLE`.
  - if `m = 0`, print an empty line (the empty string reconstructs the empty multiset).
- Time limit: 1 second. Memory: 256 MB.

Example. Input
```
3 4
aab
aba
bab
baa
```
A valid output is `babaab`: its length-3 windows are `bab, aba, baa, aab`, which is exactly the input
multiset. (`aabab` would also be accepted if it matched; here it does not, but other orderings of the
same Euler trail can.)

## Background

Treat the reconstruction as a chaining problem: two k-mers can sit next to each other iff the
`(k-1)`-character suffix of the left one equals the `(k-1)`-character prefix of the right one (they
**overlap** by `k-1`). A reconstruction is an ordering of the whole multiset in which every adjacent
pair overlaps; reading off the first k-mer and then one new character per subsequent k-mer yields the
string.

Two framings are on the table before committing:

- **Search / DP over orderings (the overlap graph).** Build a graph whose *vertices* are the k-mers
  and whose edges connect k-mers that overlap, then look for a path that visits every vertex exactly
  once. Visiting every vertex once is a **Hamiltonian path**, and finding one is NP-hard in general;
  even with pruning the worst case is exponential, which cannot survive `m = 2*10^5`.
- **A graph where the k-mers are the edges (the de Bruijn graph).** Make the *vertices* the distinct
  `(k-1)`-mers and turn each k-mer into a directed edge from its prefix `(k-1)`-mer to its suffix
  `(k-1)`-mer. Now using every k-mer exactly once is visiting every *edge* exactly once — an
  **Eulerian trail** — and that has a linear-time solution.

The whole difficulty of the problem is recognizing that the second framing is available, because the
first is the one the phrase "reassemble the fragments" suggests.

## Evaluation settings

Judged on hidden tests covering: a single k-mer; an empty list (`m = 0`); repeated k-mers /
parallel edges (e.g. `aa, aa` → `aaa`); Eulerian *circuits* where no fragment is a distinguished
start (e.g. `ab, bc, ca`); genuinely unreconstructable inputs from both degree imbalance and
disconnection (e.g. `ab, cd`); branching de Bruijn graphs where a naive greedy chaining strands edges;
and large instances with `m = 2*10^5` (both near-path graphs with `k` up to 30 and dense graphs over
a small alphabet with `k` small, so many parallel edges and a deep Euler trail).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int k, m;
    if (!(cin >> k >> m)) return 0;
    vector<string> kmers(m);
    for (auto &s : kmers) cin >> s;

    // TODO: reconstruct one string whose length-k windows are exactly the
    // given multiset of k-mers, or decide that none exists.
    string answer;

    cout << answer << "\n";
    return 0;
}
```
