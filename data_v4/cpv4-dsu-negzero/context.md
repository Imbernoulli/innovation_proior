# Richest allied guild after a stream of alliances

## Research question

A game world has `n` players, numbered `1..n`. Player `i` starts alone in their own guild and carries a
gold balance `g[i]`, which **may be negative** (debt), zero, or positive. A chronological stream of `q`
events then arrives, of two kinds:

- `1 u v` — players `u` and `v` swear an alliance, so their two guilds **merge** into one. The merged
  guild's balance is the **sum** of the two guilds' balances (debts and credits cancel). If `u` and `v`
  are already in the same guild, the event changes nothing.
- `2` — a reporter asks: **among all guilds that currently contain at least two players, what is the
  largest guild balance?** If that largest balance is **not strictly positive** — or if no guild yet has
  two or more players — the reporter prints `0` (there is no treasury worth announcing).

Process the events in order and answer every type-`2` query. The "at least two players" rule and the
"only if strictly positive, else `0`" rule are the whole subtlety: a guild can be *large* yet *broke*,
and merging a rich guild with a deeply indebted one can drop the best balance below zero.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `q` (`0 <= n <= 2*10^5`, `0 <= q <= 2*10^5`).
  The second line (present only when `n >= 1`) has `n` integers `g[1..n]` (`-10^9 <= g[i] <= 10^9`).
  Then `q` lines follow, each an event: `1 u v` (`1 <= u, v <= n`) or `2`.
- Output (stdout): for each type-`2` event, one line with the answer described above.
- Time limit: 1 second. Memory: 256 MB.

Example: see the worked sample at the end of this file; its answers are `14`, `11`, `0`, `0`.

## Background

This is a union–find (disjoint-set union) problem dressed as a guild simulator. Each guild is a connected
component; a `1 u v` event is a `union`, and a `2` event reads a per-component aggregate (the balance sum).
Two design questions sit on top of the plain DSU:

- **Per-component aggregate.** The component's balance is the sum of its members' balances. Storing that
  sum on the root and adding the absorbed root's sum on each union keeps it `O(1)` per merge. Because
  balances can be negative, the sum is **not** monotone across merges — joining a debt-laden guild can
  *lower* it — so anything that assumes "the answer only ever grows" is wrong.
- **The query over multi-member components.** A naive idea is to keep a single running maximum updated at
  merge time. That breaks precisely because of negatives: the merge that produces the current best can
  later be absorbed into a larger, poorer guild, leaving a stale maximum that no live guild attains. The
  honest approach maintains the *exact* multiset of current multi-member balances.

## Evaluation settings

Judged on hidden tests covering: all-positive balances, balances mixing negatives and zeros, the empty
world (`n = 0`), a single player (`n = 1`, which can never form a two-member guild), guilds whose balance
is driven below zero by absorbing a large debt, a balance that lands exactly on `0` (must be reported as
`0`, since the rule is *strictly* positive), self-unions and repeated unions of already-merged players,
and large `n = q = 2*10^5` with `|g[i]|` near `10^9` (so a guild balance can reach `~2*10^14` and must be
held in 64-bit).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;          // n = 0 / empty input
    vector<long long> bal(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> bal[i];

    // DSU over players; each root carries its guild's balance sum and member count.
    // TODO: handle '1 u v' as a union (sum balances, update sizes) and answer
    // each '2' with the largest multi-member balance if it is strictly positive,
    // else 0. Beware: balances are signed, so merge-time maxima go stale.

    return 0;
}
```
