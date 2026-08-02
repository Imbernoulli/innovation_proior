# Rollout Order Under Flag Dependencies

## Problem
A service has N feature flags, numbered 1..N, each with a business value v_i > 0.
You are given exactly N rollout windows — one action per window. In each window
you take ONE of three actions:

- `E x` — enable flag x. Flag x must never have been touched (enabled or rolled
  back) before this window. Every flag that x REQUIRES must currently be active.
  Flag x must not CONFLICT with any currently active flag.
- `R x` — roll back (disable) flag x. Flag x must currently be active. Once
  rolled back, a flag can never be enabled again for the rest of the schedule.
- `P` — pass; do nothing this window.

Two kinds of pairwise rules define the flag-interaction graph:
- **REQUIRES(c, p)**: flag c may only be enabled while flag p is currently
  active.
- **CONFLICTS(a, b)**: flags a and b can never be simultaneously active.
  Enabling one while the other is active is illegal — you would first have to
  roll the blocker back, which costs an entire window and permanently
  forfeits that flag (it can never be re-enabled).

## Input (stdin)
```
N
v_1 v_2 ... v_N
R
c_1 p_1
...
c_R p_R
C
a_1 b_1
...
a_C b_C
```
`R` REQUIRES pairs (flag c requires flag p), followed by `C` CONFLICTS pairs
(unordered — a and b can never both be active). The REQUIRES relation is
acyclic.

## Output (stdout)
Exactly N lines, one action per rollout window, each of the form `E x`,
`R x`, or `P`.

## Feasibility
Every action must be legal given the state built by the preceding actions in
this schedule (see above). Any violation — a malformed line, wrong line
count, an out-of-range flag id, re-touching an already-enabled-or-rolled-back
flag, enabling a flag whose REQUIRES are unmet, enabling a flag that
conflicts with a currently active flag, or rolling back a flag that isn't
active — makes the whole submission score **0**.

## Objective
For each window t, after applying that window's action, let A_t be the set
of currently active flags. Maximize the accumulated value

```
F = sum_{t=1..N}  sum_{i in A_t} v_i
```

A flag earns its value once for every remaining window it stays active, so
both WHEN a flag is turned on and WHETHER it is ever forced into a rollback
matter for the final score.

## Scoring
The checker builds its own weak reference schedule internally — it enables
only the flags that have NO REQUIRES and NO CONFLICTS at all, in index order
— and calls its total value B. Your score is
`ratio = min(1, F / (14 * B))`, printed as `Ratio: <ratio>`. An infeasible
output scores 0.

## Constraints
1 <= N <= 12, 1 <= v_i <= 100, time limit 2-5s.

## Example (illustrative only — real cases are larger)
N = 3, values = [10, 5, 8], no REQUIRES, one CONFLICT pair (2, 3).

The checker's baseline enables only flag 1 (the only flag with zero
REQUIRES and zero CONFLICTS): windows give active-sums 10, 10, 10 → B = 30,
so matching it scores 30 / (14*30) = 0.0714.

Enabling flag 1, then flag 3 (the higher-value side of the conflicting
pair), then passing: active-sums are 10, 18, 18 → F = 46 → ratio =
46 / 420 = 0.1095.

Chasing raw value order instead — enable 1, then try to enable 2 (the next
highest value), which conflicts with nothing yet so it succeeds, then try to
enable 3, which conflicts with the now-active 2 and forces a rollback of 2
before window 3 can even attempt 3 — never gets flag 3 turned on and
permanently loses flag 2's value: active-sums 10, 15, 10 → F = 35 → ratio =
35 / 420 = 0.0833, strictly worse. Deciding, up front, which side of each
conflict to take — so that every prefix of the schedule is already a legal
configuration — is exactly the insight this problem rewards.
