# Capex Stage-Gate: Sizing Commitments by What They Teach You

## Problem

A project is built from **M** modules, numbered `1..M`, that must be built
**strictly in order** (module `j` requires `1..j-1` already built — fixed
engineering precedence). You choose how to group the modules into `K`
consecutive, non-empty **stages**: stage `k` covers modules `g[k-1]+1..g[k]`
(with `g[0]=0`, `g[K]=M`). Building stage `k` costs `F + sum(costs of its
modules)` — `F` is a fixed *mobilization overhead* charged once per stage
(more, smaller stages pay it more often; bigger stages capture that scale
economy). This cost is paid, discounted by `r^(k-1)`, when stage `k` starts.

The project has a hidden true type, Good or Bad, with prior `P(Good)=p`.
Immediately after stage `k` (for `k<K`) finishes, **each module you just
built** reports a private signal (Good/Bad) about the true type: module `j`'s
signal matches the true type independently with probability `acc[j]`
(`acc[j]` near 0.5 = the module teaches you almost nothing; near 1 = it is
highly diagnostic). Having seen every signal from stages `1..k`, you must
decide: **CONTINUE** to stage `k+1`, or **ABANDON** now. Abandoning recovers
`sigma` (a fraction in `[0,1)`) of everything spent so far — the remaining
`1-sigma` is an irreversible loss — and the project ends with no payoff. If
you reach and finish stage `K`, the project completes and pays `VG` (if
truly Good) or `VB` (if truly Bad, possibly negative), discounted by `r^K`,
on top of everything already spent.

**Objective**: choose the stage boundaries AND the abandon/continue rule to
maximize the *exact expected* net present value over the hidden type and all
possible signal outcomes.

## Input (stdin)
```
M
c[1] c[2] ... c[M]              integers
acc[1] acc[2] ... acc[M]        floats in [0.50, 0.97]
p VG VB sigma F r                floats (p,sigma in (0,1); r in (0,1])
```

## Output (stdout) — your staging + decision policy
```
K
g[1] g[2] ... g[K]               strictly increasing, g[K] = M
<decision line for checkpoint 1>
...
<decision line for checkpoint K-1>
```
Checkpoint `k` (for `k=1..K-1`) has `2^(g[k])` tokens, each `0` or `1`: the
entry at index `h` is your decision for the signal history where, for each
module `j` in `1..g[k]`, bit `(j-1)` of `h` is 1 iff module `j`'s signal was
Good (module 1 = least-significant bit). `1`=continue, `0`=abandon. If
`K=1` (full commitment) no decision lines are needed.

## Feasibility
`1<=K<=M`; boundaries strictly increasing integers in `[1,M]` with
`g[K]=M`; every decision token is exactly `0` or `1` with exactly the
required count at every checkpoint; no extra tokens. Any violation scores 0.

## Scoring
The checker recomputes your policy's exact expected NPV `V` (full
enumeration over hidden type x every module signal, weighted by the true
joint probabilities — no sampling). It also computes `B`, the expected NPV
of the trivial "one giant stage, never check in" policy on the same
instance (guaranteed positive by construction). Score:
```
ratio = min(1.0, V / (10 * B))
```
So exactly matching the full-commitment baseline scores `0.1`; you need
roughly `10x` the naive baseline's value to reach `1.0`.

## Constraints
`3<=M<=7`, costs are positive integers `<=50`, time limit 5s.

## Example (illustrative, not a real test case)
`M=2, c=[2,10], acc=[0.9,0.5], p=0.5, VG=100, VB=-50, sigma=0.2, F=1, r=1.0`.
Full commitment (`K=1`) pays `13` total and nets `0.5*(100-13)+0.5*(-50-13)
= 12 = B`. A policy that splits `g=[1,2]`, always continues after module 1's
(uninformative-if-Bad-looking) signal is Good and abandons when it is Bad,
realizes expected value `34.3` — worth checking whether module 1's own
accuracy (0.9, not module 2's 0.5) makes that split actually pay for the
extra mobilization overhead on YOUR instances (it does not always: with
`acc=0.5` everywhere, splitting only adds overhead for zero information and
full commitment is optimal).
