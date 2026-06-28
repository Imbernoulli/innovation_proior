# Divisor Nim — who wins the proper-divisor take-away game

## Research question

There are `n` piles of stones; pile `i` holds `a[i]` stones. Two players alternate moves.
A single move is: pick any pile whose current size `x` satisfies `x > 1`, and replace that
pile with a pile of `y` stones where `y` is a **proper divisor** of `x` — that is, `y` divides
`x` and `1 <= y < x`. A pile of size `1` admits no move (its only divisor `< 1` does not exist).
A player who cannot move (every pile has size `1`, or there are no piles) **loses**; this is
normal play convention. The first player to move is "First".

Given the initial pile sizes, decide who wins under optimal play: print `First` if the player
who moves first wins, otherwise print `Second`.

The trap is that the reachable game-state space is astronomically large: each pile can be driven
down through its divisor lattice, and the *combined* state is the product over piles. A direct
minimax / memoized game-tree search over the joint state is exponential in the number of piles and
cannot survive `n` up to `10^5`. The question is whether the game has exploitable structure.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 10^5`), the number of piles. Then `n`
  integers `a[i]` (`1 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line, `First` or `Second`.
- Time limit: 2 seconds. Memory: 256 MB.

Example 1: `n = 3`, piles `8 12 6` → output `First`.
Example 2: `n = 2`, piles `4 9` → output `Second`.
Example 3: `n = 0` (no piles) → output `Second` (First cannot move and loses immediately).

## Background

This is an impartial combinatorial game (both players have the same moves available; the moves
depend only on the position, not on whose turn it is) played under normal-play convention. Two
facts from combinatorial game theory are relevant before committing to an algorithm:

- **Independence of piles.** A move touches exactly one pile and never interacts with the others.
  So the whole position is a *disjunctive sum* of one-pile games. The Sprague–Grundy theorem says
  every impartial game position has a Grundy (nim-) value, and the value of a sum of games is the
  bitwise XOR of the component values; the position to move from is losing for the mover iff that
  XOR is `0`.
- **The per-pile game.** A single pile of size `x` is itself an impartial game whose options are
  the proper divisors of `x`. Its Grundy value is `mex` (minimum excludant) of the Grundy values of
  its options. Whether this per-pile value has a closed form — and what it is — is the crux.

A foil worth keeping in mind: treating the piles as ordinary Nim and XOR-ing the *sizes* `a[i]`
(or XOR-ing `a[i]-1`, etc.) is wrong, because a move here is "replace `x` by a divisor", not
"remove any number of stones". The correct per-pile value must be derived, not assumed.

## Evaluation settings

Judged on hidden tests covering: `n = 0`; single pile of size `1` (immediate loss for First);
single prime, prime powers (`2^k`, `3^k`, …), and highly composite values; piles that XOR to zero
vs nonzero; values up to `10^9` forcing real factorization; and large `n = 10^5` with many large
values (so per-pile work must be near `O(sqrt(a))` or better). The reference is a full-minimax
oracle on small inputs (no Sprague–Grundy assumed), cross-checked against an independent
factorization on large values.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: decide the winner of the proper-divisor take-away game on these piles.
    bool firstWins = false;

    cout << (firstWins ? "First" : "Second") << "\n";
    return 0;
}
```
