# One Passphrase, Many Gatekeepers

## Problem
A vault is guarded by `m` independent gatekeepers sharing one alphabet of `A` symbols
`{0, 1, ..., A-1}`. Gatekeeper `i` runs a deterministic finite automaton (DFA): `n_i`
states, a fixed start state, a set of accepting states, and a complete transition
table (next state for every state and every symbol). Gatekeeper `i` also carries an
integer **weight** `w_i`.

You submit **one** passphrase `S` (a string over the shared alphabet, length between
`0` and `Lmax`). Each gatekeeper reads the *same* `S` from its own start state,
symbol by symbol, and accepts iff it ends in one of its own accepting states. You
collect the weight of every gatekeeper that accepts. The instance is built so that
**no single passphrase is accepted by all `m` gatekeepers at once** — some gatekeepers
are fundamentally incompatible with each other, and part of the problem is figuring
out which ones to give up on.

## Input (stdin)
```
m A Lmax
```
then, `m` times, a gatekeeper block:
```
n start w
k a_1 a_2 ... a_k
n lines, each with A integers: transition row for states 0..n-1 (next state per symbol)
```
`n` = number of states, `start` = start state, `w` = weight (integer, `w >= 1`),
`k` and `a_1..a_k` = the accepting states. Row `t` of the transition table gives, for
each symbol `0..A-1`, the state reached from state `t` on that symbol.

## Output (stdout)
```
L
s_1 s_2 ... s_L
```
`L` is the length of your passphrase (`0 <= L <= Lmax`); the second line lists its
`L` symbols (each in `[0, A-1]`). If `L = 0` the second line may be empty.

## Feasibility
Output must contain **exactly** `L + 1` whitespace-separated integer tokens: `L`
itself, then exactly `L` symbols. Any of the following scores `Ratio: 0.0`: a
non-integer token, `L` outside `[0, Lmax]`, a symbol outside `[0, A-1]`, or any
extra/missing tokens.

## Objective
Simulate every gatekeeper on `S`. Let `F` be the sum of `w_i` over gatekeepers that
accept `S`, minus a tiny tie-break `0.0001 * L` that favors a shorter passphrase
when two candidates collect the exact same total weight. **Maximize `F`.**

## Scoring
The checker independently builds its own trivial reference: for each gatekeeper it
finds (by BFS on that gatekeeper's own graph, ignoring all others) the shortest
passphrase — if any — that satisfies *that gatekeeper alone*; `B` is the largest
weight among gatekeepers with such a reachable solo passphrase. With `F` from your
submission:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Satisfying only the single heaviest gatekeeper (and no one else) scores `Ratio ~=
0.1` (exactly `0.1` minus the negligible length tie-break); collecting `10x` that
weight caps the score at `1.0`.

## Constraints
- `4 <= m <= 8`, `A = 3`, `10 <= Lmax <= 23`.
- Every gatekeeper has at least one reachable accepting state within `Lmax` steps.
- No passphrase of length `<= Lmax` is accepted by all `m` gatekeepers simultaneously.
- Time limit 5s, memory 512MB.

## Example
Alphabet `A = 2`. Three gatekeepers, `Lmax = 4`:
- Gatekeeper 1 (`w=10`): 4 states, start 0, accept `{2}`; from 0 symbol 1 -> state 1
  (else -> a dead state 3 that never accepts), from state 1 symbol 0 -> state 2
  (else -> dead). So it accepts exactly passphrases starting `"10"`.
- Gatekeeper 2 (`w=6`): accepts exactly passphrases starting `"00"`.
- Gatekeeper 3 (`w=6`): accepts exactly passphrases starting `"000"`.

No passphrase can start with both `"10"` and `"00"`, so the full 3-way intersection
is empty (as guaranteed by the input). The solo baseline is `B = 10` (gatekeeper 1
alone). Submitting `S = "10"` satisfies only gatekeeper 1: `F ~= 10 - 0.0002 = 9.9998`,
`Ratio ~= 0.09998` (about the trivial reference). Submitting `S = "000"` instead
satisfies gatekeepers 2 **and** 3 together: `F ~= 12 - 0.0003 = 11.9997`, `Ratio ~=
0.11999` — worse than a smarter choice elsewhere in a larger instance, but strictly
better than fixating on the single heaviest gatekeeper. (This tiny example is
illustrative only; real instances plant much larger weight gaps.)
