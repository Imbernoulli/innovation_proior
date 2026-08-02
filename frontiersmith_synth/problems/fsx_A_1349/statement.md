# The Smallest Machine That Does The Job

## Problem

A **one-way machine** reads a string of bits `b0 b1 ... b(L-1)` left to right,
never stepping back. It has states numbered `0 .. S-1` (you choose `S`),
always starting in state `0`. At every step it reads the symbol at the
current tape position — `0`, `1`, or, once the string is exhausted, an
infinite run of the blank symbol `_` — and deterministically either moves to
another state and advances one position, or **halts** in a special `ACCEPT`
or `REJECT` outcome (`ACCEPT`/`REJECT` are not counted among your `S`
states; they are free, mandatory halting outcomes). The machine must halt
within a given step bound; if it never reaches `ACCEPT`/`REJECT` within that
bound, its verdict on that string is undefined (wrong).

A fixed, unknown, deterministic rule classifies every finite bitstring as
`0` (reject) or `1` (accept). You are given a sample of strings the rule has
already labelled. Your machine must reproduce the rule **exactly** — not
just on the strings you were shown, but on every string the checker probes
it with afterwards (up to length 250) — using as **few states as possible**.
Many machines can pass the sample you see; only some of them capture the
actual rule rather than an artifact of that particular sample, and among
the machines that do capture it correctly, some throw away far more
redundant bookkeeping than others.

## Input (stdin)

```
SEED STEP_BOUND
N
bits_1 label_1
bits_2 label_2
...
bits_N label_N
```
`bits_i` is a nonempty string over `{0,1}`; `label_i` is `0` or `1`. `SEED`
is an opaque instance tag (not needed for your solution). `STEP_BOUND` is
the maximum number of machine steps allowed before a run is judged
non-halting.

## Output (stdout)

```
S
t0_0 t1_0 tb_0
t0_1 t1_1 tb_1
...
t0_{S-1} t1_{S-1} tb_{S-1}
```
Line `1+s` (for state `s`, `0 <= s < S`) gives the three transitions out of
state `s`: on reading `0`, on reading `1`, on reading blank `_`. Each
`tX_s` is either an integer in `[0,S)` (move to that state, advance one
position) or the literal `A` / `R` (halt ACCEPT / REJECT immediately).

## Feasibility

Your machine is feasible iff, simulated from state `0` for at most
`STEP_BOUND` steps, it halts with the correct verdict on **every** given
sample string **and** on every hidden probe string the checker generates
from the same rule. Any wrong verdict, any run that fails to halt within
the bound, or any malformed output (bad token, out-of-range target,
`S<1`) scores `Ratio: 0.0`.

## Objective

Minimize `S`, the number of non-halting states, subject to feasibility.

## Scoring

The checker independently reconstructs the ground-truth rule from the
sample pairs alone (it is fully determined by them) and, from it, builds
its own **correct but deliberately redundant** reference machine with `B`
states. Your feasible machine's state count `F = S` scores
```
sc = min(1000, 100 * B / F);  Ratio = sc / 1000
```
Smaller `F` (relative to `B`) scores higher, capped below `1.0` — matching
the checker's own construction exactly earns `0.1`; using noticeably fewer
states earns proportionally more, but the cap keeps headroom above any
reference solution.

## Constraints

`1 <= N <= 1000`, `1 <= len(bits_i) <= 60`, `1 <= S <= 20000`,
`STEP_BOUND <= 400`. Time limit 5s, memory 512MB.

## Example (worked score, illustrative numbers only)

Suppose the checker's reference machine for some instance uses `B=32`
states (a correct but padded encoding). A submission using `F=32` states
scores `Ratio = min(1000, 100*32/32)/1000 = 0.1`. A submission that finds
an equivalent machine using only `F=4` states (because it recognized that
28 of the 32 states never actually behave differently from one another)
scores `Ratio = min(1000, 100*32/4)/1000 = 0.8`. A submission whose machine
gets even one hidden probe string wrong, or loops without ever halting,
scores `Ratio = 0.0` regardless of how few states it used.
