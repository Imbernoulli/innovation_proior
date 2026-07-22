# March of 64: Bitsliced FSM Transition Synthesis

## Problem
A tiny deterministic finite automaton has `S` states (`0..S-1`) and an alphabet
of `K` symbols (`0..K-1`), given by a total transition table `delta[s][k]`.
`S` and `K` are exact powers of two: `S = 2^b`, `K = 2^m`.

You are building a SIMD "bitslice" engine that advances **64 independent
instances of this automaton in lockstep, one step at a time**. Each instance
lives in its own lane of a set of 64-bit words: one word per state-bit-plane,
one word per input-bit-plane. Lane `j` of plane `i` holds bit `i` of instance
`j`'s current state/input. Because every lane runs the identical program on
its own private data, your program may only use **bitwise, lane-local** gates
`AND`, `OR`, `XOR`, `NOT` — never anything that mixes bits across lanes.

The catch: which `b`-bit code represents which state is entirely up to you.
The public numbering `0..S-1` is arbitrary — you may relabel the states with
any bijection to `{0,1}^b` before building your circuit. Under a well-chosen
relabeling, most of the transition function's dependence on the state and
symbol bits collapses to XORs (a few points may still resist any single
relabeling and need extra AND-correction terms); a careless relabeling (e.g.
keeping the given numbering) leaves the same dynamics looking dense and
forces a much bigger generic circuit for the identical automaton.

## Input (stdin)
```
S K
```
then `S` lines of `K` integers: row `s` lists `delta[s][0] .. delta[s][K-1]`.
Then 64 lines, each `s k`, giving the (state, symbol) pair loaded into that
lane before this step (only used by the checker to pick which lanes to test —
your circuit must be correct for every reachable `(state,symbol)` pair, since
between them the 64 lanes cover the whole table).

## Output (stdout)
```
b m
S
s_0 code_0
...
s_{S-1} code_{S-1}
P
<P gate lines>
o_0 o_1 ... o_{b-1}
```
- `b m`: must equal the true `b = log2(S)`, `m = log2(K)`.
- `S` lines: your state encoding, a permutation — every state `0..S-1` once,
  every code `0..S-1` used exactly once (a bijection onto the `b`-bit codes).
- Wires are numbered from 0: wires `0..b-1` hold bit `i` of the **current**
  state's code; wires `b..b+m-1` hold bit `i` of the current symbol (standard
  binary of the symbol's own integer value). Wire `b+m+t` is the output of
  gate line `t` (0-indexed), one of `AND a c`, `OR a c`, `XOR a c` (binary) or
  `NOT a` (unary), where `a`,`c` are indices of *already-defined* wires only
  (no forward references, no cycles).
- `P`: number of gate lines, followed by exactly `P` lines.
- Final line: `b` wire indices — the **new** state's code, bit `i` on wire
  `o_i`. Reconstruct it as an integer and look up which state your encoding
  gave that code; that is the produced next-state.

## Feasibility
Reject (score 0) on: `b,m` or state-count mismatch; the encoding not covering
every state exactly once or not a bijection onto `0..S-1`; any malformed /
out-of-range / non-finite token; any forward-referencing or unknown gate; or
if the simulated circuit's next-state disagrees with `delta[s][k]` for **any**
of the 64 given lanes.

## Objective
Minimize `P`, the total gate count (scalar bitwise operations), for the fixed
64-lane word width.

## Scoring
The checker builds its own naive baseline `B`: for every `(state, symbol)`
pair in the identity numbering, an AND-indicator built completely from
scratch (a fresh NOT gate per negated literal, no sharing across indicators),
then one OR-select per output bit — a valid but wasteful circuit. With your
gate count `P`:
```
Ratio = min(1, 0.1 * B / P)
```
Fewer gates is better; matching `B` scores `0.1`; a tenth of `B` caps at `1.0`.

## Constraints
`4 <= S <= 8`, `2 <= K <= 4` (both exact powers of two); `0 <= P <= 20000`.
Deterministic, exact integer/bitwise simulation; no timing, no randomness in
scoring. Time limit 5s.

## Example
`S=4,K=2`, `delta = [[1,2],[3,0],[0,1],[2,3]]`. Under identity encoding
(`code(s)=s`) this happens to already be XOR-like, needing few gates; on most
real instances the "obvious" identity numbering hides a messier dependency,
and the naive decoder (`B`) needs tens of gates where a well-chosen
relabeling needs only a handful. (Illustrative only — real instances are
generated so identity is generally NOT the best relabeling, and no
relabeling is guaranteed to make the whole table gate-free.)
