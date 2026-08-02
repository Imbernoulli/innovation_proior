# Three Calls Deep: Scheduling a Dependency-Aware Fuzzer

You are testing a small stateful API. It exposes **T** resource **types**,
labelled `0..T-1` in no particular meaningful order. Two kinds of call:

- `CREATE t` — attempt to make a fresh resource of type `t`. It succeeds
  (binding a brand-new resource id to this call) **iff every type in
  `deps[t]` already has at least one successfully-created resource**
  (from any earlier call, of any of your calls so far). Otherwise the call
  is simply wasted — no resource, no error, nothing happens.
- `OP r c` — apply opcode `c` (`0 <= c < C`) to the resource created by
  your call-line `r` (`r` is the 0-indexed position of that earlier line
  in *your own* sequence). If line `r` never produced a live resource
  (its `CREATE` failed, or `r` refers to an `OP` line, or `r >= ` your
  current position), this is also just wasted. If it did, the resource's
  local state transitions per that type's own transition table — some
  opcodes advance it, others leave it exactly where it was.

Every resource starts at local state `0`. Each type `t` has its own state
count `S_t` and an `S_t x C` transition table telling you, for every state
and opcode, the next state. **Deeper states are worth quadratically more**
than shallow ones — your total score is the sum, over every `(type,
state)` pair *any* resource of yours ever reached (state `>= 1`, first
time only), of `state^2`. Reaching state 3 of some type nets `9`; reaching
states `1,2,3` in order (you must pass through them) nets `1+4+9=14`.
Some types need a very particular multi-step opcode sequence to make any
progress at all — a wrong opcode simply does not advance that state.

You have a fixed call **budget** (`CREATE`+`OP` calls combined,
`0 <= your total <= budget`). Endpoints that look independent are often
not: most valuable states sit behind resources whose `deps[]` chain must
be built up first, and the schema does not tell you which types are
"foundational" — that is exactly what you must read out of `deps[]`
yourself; label order carries no information about dependency depth.

## Input (stdin)

```
T C budget
```
then, for each type `t = 0..T-1` in label order:
```
k dep_1 ... dep_k
S_t
S_t lines, each C integers: trans[t][s][c] = next state for opcode c
```
`1 <= T <= 12`, `2 <= C <= 4`, `1 <= S_t <= 4`, all `dep_i < T`. `deps[]`
forms a DAG (no cycles).

## Output (stdout)

```
M
```
then `M` lines, each `C t` or `O r c`. `0 <= M <= budget`. All tokens are
read positionally (whitespace/newlines both fine): `t` must satisfy
`0 <= t < T`; for `O r c`, `0 <= r < M` and `0 <= c < C`.

## Scoring

The checker replays your sequence exactly as specified above, strictly
validating every token (any malformed/out-of-range token, or wrong total
token count, scores `Ratio: 0.0`). Then:

```
F = sum over every (type, state>=1) pair ever reached of state^2
```

against an internal baseline `B`: the best score reachable by touching
*only* the types with `deps[t] == []` (create it, apply whichever single
opcode looks best) — the "don't even think about the dependency graph"
reference, always positive since some type always has empty `deps`.

```
Ratio = min(1.0, 100 * F / (10 * B)) / 100    -- printed to 6 decimals
```

Matching `B`'s quality scores `~0.1`; genuinely exploiting the dependency
structure and the transition tables scores well above that.

## Worked example (illustrative form only — not the scoring input)

`T=2, C=2, budget=4`. Type `0`: `deps=[]`, `S=2`,
`trans[0]=[[1,1],[1,1]]` (either opcode advances `0->1`). Type `1`:
`deps=[0]`, `S=2`, `trans[1]=[[1,0],[0,1]]` (only opcode `0` advances).
Output `M=4 / C 0 / O 0 0 / C 1 / O 2 0` creates type 0, advances it to
state 1 (`+1`), creates type 1 (deps satisfied — type 0 exists), advances
it to state 1 with opcode 0 (`+1`). `F = 1 + 1 = 2`.

## Constraints

Time limit 5 s, memory 512 MB. All arithmetic is exact integers;
deterministic given the input.
