# A Handshake That Must Refuse Every Forbidden Order

## Problem

A protocol handshake is a sequence of message symbols drawn from an alphabet
`{0, 1, ..., m-1}`. You are given a set of **legal traces** `L` (complete handshakes
that a correct implementation must accept) and a set of **forbidden traces** `F`
(downgrades, replays, and out-of-order/rollback attempts that a correct implementation
must reject). Both sets are finite and disjoint.

Your task: output a deterministic finite automaton (DFA) over the given alphabet that
accepts every trace in `L` and rejects every trace in `F`, using as **few states as
possible**. The automaton always starts in state `0`.

Any DFA that gets this right is *feasible*. Among feasible DFAs, smaller is better —
the whole point of a handshake state machine is to be a small, auditable object, not a
giant table that merely memorizes every string it was shown.

## Input (stdin)

```
m
L
len_1 s_1 s_2 ... s_len_1
...                                  (L lines, the legal traces)
F
len_1 s_1 s_2 ... s_len_1
...                                  (F lines, the forbidden traces)
```
`m` is the alphabet size. Each trace line begins with its length, followed by that many
symbols in `{0, ..., m-1}`. Traces have length >= 1. `1 <= m <= 8`, `L, F <= 40`, no
trace longer than 20 symbols.

## Output (stdout)

```
N
b_0 b_1 ... b_{N-1}
d_0,0 d_0,1 ... d_0,m-1
...
d_{N-1},0 ... d_{N-1},m-1
```
`N` is your state count (`1 <= N`). The second line gives the accept bit (`0` or `1`)
of every state. Each of the next `N` lines gives, for state `i`, the target state
`d_i,c` for every symbol `c = 0..m-1` — this must be a **total** function: every
state needs a defined transition for every symbol, including symbols that never occur
in any given trace at that point. State `0` is the start state.

## Feasibility

Run every trace in `L` from state `0`, following your transitions symbol by symbol; the
final state must have accept bit `1`. Run every trace in `F` the same way; the final
state must have accept bit `0`. If any trace is misclassified, or the output is
malformed (wrong token count, out-of-range state index, non-finite/non-integer token),
the score is **0**.

## Scoring

Feasible submissions are scored against the checker's own reference construction `B`:
an *unminimized* trie built from the legal traces plus one explicit catch-all reject
state for every symbol that does not extend a legal prefix (this reference is always
feasible, for any `L`, `F`). Your score is

```
ratio = min(1000, 100 * B / N) / 1000
```

i.e. matching the reference's state count scores `0.1`; using `10x` fewer states or
better saturates at `1.0`. There is no known closed form for the true minimum on a
given instance beyond exhaustive search; **exploit the structure of the given traces**.

## Constraints

`1 <= m <= 8`; `1 <= |L|, |F| <= 40`; trace length `1..20`; every symbol in
`0..m-1`; `L` and `F` are disjoint as full strings. Time limit 5s, memory 512MB.

## Example (illustrative only, not a real test case)

Say `m=2`, `L = {"01"}`, `F = {"00"}`. The reference construction (trie over `L`
plus a sink) needs 4 states: `root -0-> a`, `a -1-> accept`, and every other
transition (including `root`'s own `1`-edge and `a`'s `0`-edge) into a fourth,
self-looping reject state — so `B = 4`.

A feasible 3-state DFA: state `0` (start, non-accepting): `0->1`, `1->0`. State `1`
(non-accepting): `0->0`, `1->2`. State `2` (accepting): `0->2`, `1->2`. Trace `"01"`
runs `0 -0-> 1 -1-> 2` (accept, correct). Trace `"00"` runs `0 -0-> 1 -0-> 0`
(non-accept, correctly rejected) — by reusing the *start* state as the reject target
for `1`'s stray `0`-edge instead of allocating a brand-new sink, no legal trace is
harmed and no forbidden trace slips through. Score: `min(1000, 100*4/3)/1000 =
0.1333`. Reusing an existing state safely (instead of habitually creating a fresh sink,
or worse, resetting to the *start* state in a way that lets an attacker resume with a
fresh legal trace) is exactly the kind of structural reuse this problem rewards.
