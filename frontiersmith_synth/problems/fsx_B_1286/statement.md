# Deep-Tier Blind Spot: Map Before You Audit

## Problem
A focal firm has a 3-tier supplier forest. **Tier-1** nodes are its direct
suppliers and are visible from the start. Each tier-1 node has some
**tier-2** sub-suppliers, and each tier-2 node has some **tier-3**
sub-suppliers (leaves) — but a tier-2 or tier-3 node is *invisible* (you
don't even know it exists) until its parent has been **mapped**.

Every node `v` carries a raw ESG risk value `risk[v]`. Because risk
propagates upstream through the chain, a node's risk counts toward the
focal firm's total exposure discounted once per tier hop by a fixed
propagation factor `PROP` (0<PROP<1): a tier-`t` node's discounted exposure
is `risk[v] * PROP^(t-1)`. There are far more tier-3 nodes than tier-1
nodes, so their *aggregate* discounted exposure can still dominate the
total even though each individual one counts for less.

You have a fixed integer **budget**. You spend it on two kinds of actions:
- **`M v`** (map `v`): only legal for a tier-1/tier-2 node that is currently
  visible and not yet mapped; costs `mapcost[v]`. It reveals all of `v`'s
  children (making them visible for future actions) and gives `v` itself a
  partial mitigation `MAPMIT` fraction of its discounted risk (visibility /
  monitoring alone reduces risk somewhat).
- **`A v`** (audit `v`): only legal for a currently visible node not yet
  audited; costs `auditcost[v]`. It fully mitigates `v`'s discounted risk by
  fraction `AUDITMIT` (`AUDITMIT > MAPMIT`), overriding any map mitigation.

An unmapped, unaudited node contributes its full discounted risk to the
exposure — **unmapped risk cannot be audited at all**, no matter how large.

## Input (stdin)
```
N T1 BUDGET
PROP MAPMIT AUDITMIT
id_1 tier_1 parent_1 risk_1 mapcost_1 auditcost_1
...
id_N tier_N parent_N risk_N mapcost_N auditcost_N
```
`N` = total nodes, `T1` = number of tier-1 nodes (informational), `BUDGET`
integer. Each node line: `tier` in {1,2,3}; `parent=0` for tier-1, else the
id of its (tier-1 or tier-2) parent, listed earlier in the file. Tier-3 rows
have `mapcost=0` (mapping a leaf is illegal — it has no children). Ids run
1..N, tier-1 first.

## Output (stdout)
Zero or more lines `M v` / `A v`, one action per line, in the order you want
them applied. Nothing else.

## Feasibility
Every action must be legal at the moment it is applied (target visible,
not already mapped/audited as appropriate for that op, tier-1/2 only for
`M`), and the running total of action costs must never exceed `BUDGET`. Any
violation scores 0 for the whole submission.

## Objective
Maximize total discounted-risk exposure reduced:
`sum over audited v of AUDITMIT*risk[v]*PROP^(tier[v]-1)`
`+ sum over mapped-but-not-audited tier-1/2 v of MAPMIT*risk[v]*PROP^(tier[v]-1)`.

## Scoring
The checker computes your reduction `F` and an internal baseline `B` (the
value of auditing only the single highest-raw-risk tier-1 supplier — the
obvious first move). Score `= min(1, F / (10*B))`, so matching `B` scores
0.1 and reaching `10*B` saturates at 1.0.

## Example (illustrative shape only, not the hidden law)
2 nodes, `PROP=0.5, MAPMIT=0.2, AUDITMIT=0.8`, tier-1 node `v` (risk 10,
auditcost 4), tier-2 node `w`, child of `v` (risk 20, auditcost 3,
mapcost 2), `BUDGET=6`. Baseline `B` = audit `v` alone = `0.8*10=8`.
Submission `M v` then `A w` costs `2+3=5<=6`: `v` gets `MAPMIT` mitigation
`0.2*10=2`, `w` becomes visible and audited: `0.8*20*0.5=8`. `F=2+8=10`,
score `=min(1,10/80)=0.125` — beating the tier-1-only baseline by spending
part of the budget on visibility instead of a bigger direct audit.

## Constraints
1 <= N <= 200, small positive integer costs/risks, `BUDGET` fits in a
32-bit int. Time limit 5s.
