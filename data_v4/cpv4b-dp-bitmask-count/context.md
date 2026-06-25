# Counting guild-team partitions under feuds and size limits

## Research question

A guild has `n` adventurers, numbered `0..n-1`. The guildmaster wants to split **all** of them into
one or more **teams** so that every adventurer belongs to exactly one team — that is, the teams form a
*set partition* of the adventurers. A partition is **valid** only if every team satisfies two rules:

- **Size rule.** Each team has between `L` and `R` members, inclusive.
- **Feud rule.** Some pairs of adventurers feud and refuse to share a team. A team is allowed only if
  it contains no feuding pair.

Teams are **unlabeled**: a partition is a *set of teams*, so `{ {0,1}, {2,3} }` and
`{ {2,3}, {0,1} }` are the **same** partition and must be counted once, not twice.

Count the number of valid partitions, modulo `998244353`. (If no valid partition exists, the answer is
`0`. The empty guild `n = 0` has exactly one partition — the empty one.)

This is a counting/constructive variant of subset DP. The whole difficulty is *counting each partition
exactly once*: the obvious "pick a first team, recurse on the rest" recurrence counts **ordered**
sequences of teams and over-counts every partition by the number of orderings of its teams. Getting
the de-duplication right — and the modular arithmetic — is the point.

## Input / output contract

- Input (stdin), all whitespace-separated:
  - first line: four integers `n L R m` with `0 <= n <= 16`, `1 <= L <= R <= 16`, and
    `0 <= m <= n*(n-1)/2`;
  - then `m` lines, each two integers `u v` (`0 <= u < v < n`) meaning `u` and `v` feud.
    Each unordered pair appears at most once.
- Output (stdout): a single line — the number of valid partitions modulo `998244353`.
- Time limit: 1 second. Memory: 256 MB.

Example: `n = 4`, `L = 1`, `R = 2`, feuds `{0,1}` and `{2,3}`. The answer is `7` (worked below).

## Background

The constraint that teams are *unlabeled* turns this into a partition-counting problem rather than an
assignment problem. Two framings are on the table before committing:

- **Inclusion over an ordered DP.** Define `g[mask]` = number of ways to write the set `mask` as an
  *ordered* list of legal teams, via `g[mask] = sum over legal teams S ⊆ mask of g[mask\S]`. This is
  easy to write, but it counts a partition into `k` teams `k!` times, and dividing by `k!` afterwards
  is awkward because different partitions have different `k`. The open question is whether there is a
  recurrence that counts unordered partitions *directly*.
- **Anchored subset DP.** Define `f[mask]` = number of *unordered* legal partitions of `mask`. To
  avoid order, force a canonical choice: the team containing the **lowest-indexed** element of `mask`
  is decided first. Summing only over teams `S` that contain that anchor element makes each partition
  appear exactly once. The open questions are the exact anchor bookkeeping and the legality precompute.

A team `S` is *legal* iff `L <= popcount(S) <= R` and `S` contains no feuding pair; legality of all
`2^n` subsets can be precomputed once.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (answer `1`); `n = 1`; dense feud graphs (forcing all
singletons, so the answer is `0` or `1` depending on `L`); no feuds with `L = 1, R = n` (the answer is
the Bell number `B(n)` mod `p`); tight size windows that make the partition infeasible (answer `0`);
size windows like `L = R = 2` on odd `n` (infeasible); and full `n = 16` worst cases that exercise the
`3^n` subset enumeration and the modular reduction.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353;

int main() {
    int n, L, R, m;
    if (!(cin >> n >> L >> R >> m)) return 0;

    // feud[i] = bitmask of adventurers that i refuses to share a team with.
    vector<int> feud(n, 0);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;             // 0-indexed
        feud[u] |= (1 << v);
        feud[v] |= (1 << u);
    }

    int full = (1 << n) - 1;

    // TODO: precompute team legality, then count UNLABELED legal partitions
    //       of the full set modulo MOD (each partition exactly once).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
