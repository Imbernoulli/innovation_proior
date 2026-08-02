# Paste and Fold: Chain-Aware Inlining Under an Instruction-Cache Budget

## Problem

A compiler is deciding which of `m` call sites to **inline** (paste the callee's body
directly into the caller instead of leaving a real call). Call site `i` has a dynamic
execution frequency `freq_i` (how many times it runs over the program's lifetime), a
per-invocation body cost `base_cost_i` (instructions executed inside the callee every
time it runs -- paid regardless of the decision), and an `inline_size_i` (static
instructions added to the compiled image if `i` is inlined). Leaving a call site
un-inlined costs a fixed `CALL_OVERHEAD` instructions per invocation (the call/return
sequence); inlining removes that overhead but grows the code image.

The code image also starts at a fixed `S_base` and grows by `inline_size_i` for every
inlined site. If the total image exceeds `ICACHE_CAP`, the processor starts missing in
the instruction cache and **every** executed instruction gets more expensive -- not just
the newly inlined ones.

Some call sites sit in an **unlock chain**: call site `i` has a `parent_i` (an earlier
call site, or `0` for none). If `i` is inlined, its `bonus_i` (instructions shaved off
`base_cost_i` per invocation, from constant propagation into the freshly pasted copy)
only fires if `parent_i` is ALSO inlined -- and `parent_i`'s own condition holds too, all
the way up to a call site with `parent = 0`. A break anywhere upstream means no
compile-time constant reaches `i`, so `i` runs at its plain `base_cost_i`. A chain root
(`parent_i = 0`) always has `bonus_i = 0`, but inlining it may be exactly what unlocks a
bonus further down its chain.

Your job: choose which call sites to inline to minimize total cost.

## Input (stdin)

```
m
S_base ICACHE_CAP CALL_OVERHEAD PENALTY_COEF
freq_1 base_cost_1 inline_size_1 parent_1 bonus_1
...
freq_m base_cost_m inline_size_m parent_m bonus_m
```
All values are positive integers except `parent_i` (`0 <= parent_i < i`) and `bonus_i`
(`0` when `parent_i = 0`, else `0 < bonus_i < base_cost_i`).

## Output (stdout)

```
k
i_1 i_2 ... i_k
```
`k` distinct call-site indices (1-based, any order) to inline. `k = 0` is allowed.

## Feasibility

Rejected (score 0) if: any token is missing, non-integer, or malformed; `k` is negative
or exceeds `m`; any index is out of `[1,m]`; or an index repeats.

## Scoring

Let `x_i = 1` if site `i` is inlined, else `0`. Site `i` is **unlocked** if `x_i = 1` and
(`parent_i = 0`, or `parent_i` is unlocked and `x_{parent_i} = 1`) -- i.e. the entire
chain from the root down to `i` is inlined. Then:

```
effective_cost_i = base_cost_i - bonus_i   (if unlocked)   else base_cost_i
D = sum_i freq_i * ( effective_cost_i + CALL_OVERHEAD * (1 - x_i) )
S = S_base + sum_i x_i * inline_size_i
excess = max(0, S - ICACHE_CAP)
F = D * (ICACHE_CAP + PENALTY_COEF * excess) // ICACHE_CAP     (integer floor division)
```
Minimize `F`. Let `B` be the checker's own reference construction: sort call sites by
`(freq_i * CALL_OVERHEAD) / inline_size_i` descending and greedily inline while the image
stays within `ICACHE_CAP` -- a size-aware frequency recipe that never looks at
`parent`/`bonus` at all. The score is `Ratio = min(1.0, 0.1 * B / F)`.

## Example (illustrative, not a worked score)

`CALL_OVERHEAD = 10`. Site 3: `parent_3 = 0`, `freq_3 = 5`, `base_cost_3 = 30`,
`inline_size_3 = 150` (a root; inlining it alone only saves `5*10 = 50` -- low value).
Site 5: `parent_5 = 3`, `freq_5 = 500`, `base_cost_5 = 100`, `bonus_5 = 70`.

Inline neither: `5*(30+10) + 500*(100+10) = 55200`.
Inline only site 5 (its parent stays locked): `5*(30+10) + 500*(100+0) = 50200` -- a
frequency-sorted recipe happily takes this, since site 5 alone looks like a big win.
Inline BOTH: site 5 is now unlocked, `effective_cost_5 = 100-70 = 30`:
`5*(30+0) + 500*(30+0) = 15150` -- far better, even though site 3's own standalone value
never signaled this.

## Constraints

`10 <= m <= 60`, `1 <= S_base, freq_i, base_cost_i, inline_size_i <= 5000`,
`1 <= ICACHE_CAP <= 20000`, `1 <= CALL_OVERHEAD <= 30`, `1 <= PENALTY_COEF <= 10`.
Time limit 5s, memory 512MB.
