# Inspector Who Must Guarantee, Not Guess

## Setup
A manufactured panel occupies the square `[0,S] x [0,S]`. Its nominal (as-designed) height is
a fully public, exactly-computable surface `z_nom(x,y)`, coefficients `a0..a5` given:
```
z_nom(x,y) = a0 + a1*x + a2*y + a3*x*y/S + a4*(x-S/2)^2/S + a5*(y-S/2)^2/S
```
The panel's ACTUAL height is `z(x,y) = z_nom(x,y) + g(x,y)`; the deviation field `g` is
**hidden**, globally Lipschitz with published constant `Lc` (`|g(p)-g(q)| <= Lc*dist(p,q)`),
and `|g|` never exceeds a published cap `m_cap`.

You get a fixed budget of PROBES, spent one at a time, adaptively, over several rounds. Each
probe is a point `(x,y)` you choose; it returns the EXACT `z(x,y)` there (no noise). Before
probing in round `r` you see every reading so far — including a small free "pilot" batch
already on file (round 0's history). After the budget is spent you submit a single number:
your **certificate** `bound = D`.

## The verification grid, and why the certificate must be earned, not guessed
The certificate is checked against a fixed, deterministic `n_inspect x n_inspect` grid of
inspection points spanning the panel (`x = S*(i+0.5)/n_inspect`, same for `y`) — not an
abstract continuum, so you and the grader always compute the identical point set. For any
probe `p_i` with deviation `dev_i = z(p_i) - z_nom(p_i)`, Lipschitz continuity guarantees
`|g(q)| <= |dev_i| + Lc*dist(q,p_i)` for every inspection point `q`, so the tightest bound
honestly justifiable at `q` is `min(m_cap, min_i(|dev_i| + Lc*dist(q,p_i)))`; over the WHOLE
grid it's `D_earn`, the max of that over every inspection point. **The grader recomputes
`D_earn` itself, from the ACTUAL probe history it produced for you**, the instant you submit
`D`. If `D < D_earn` (minus a tiny tolerance) your certificate is unsound and scores `0` — you
cannot "know" the answer without earning it, since `D_earn` depends only on readings the
grader itself generated. A needlessly large `D` is honest but wastes tightness already earned.
You are optimizing the worst case of this formula on your OWN eventual probe set — a moving
target, so placement is a game against the bound formula, not a checklist.

## Protocol (stdin -> stdout, called once per round)
`phase == "probe"`: stdin has `{"phase","domain":{"x_lo","x_hi","y_lo","y_hi"},"lipschitz_L",
"m_cap","n_inspect","z_nom_coeffs":{"a0".."a5"},"budget_total","budget_left","round",
"rounds_total","max_probes_this_round","history":[{"x","y","z"},...]}`. Return
`{"probes":[{"x":..,"y":..}, ...]}`, at most `max_probes_this_round` (always `1`) entries. A
point outside the domain is silently skipped (charged, no reading). A malformed entry (wrong
type, non-finite) scores the WHOLE instance `0`.

`phase == "certify"` (final call): stdin has the same fields plus the final `history`. Return
`{"bound": D}`, one finite `D >= 0`. Wrong type/shape/non-finite, or any crash, timeout, or
non-JSON output on any call, scores that instance `0`.

## Scoring
```
D_earn  = the grader's own recomputation of the tightest sound bound from your ACTUAL history
quality = clip(1 - D / D0, 0, 1)          (only if D >= D_earn - tol; else the instance is 0)
score   = 0.10 + 0.82 * quality            (clipped to [0,1])
```
`D0` is `D_earn` using ONLY the free pilot batch (zero of your own budget spent) — the
grader's zero-effort reference. Final score = mean over 10 fixed, seeded instances.

## Traps
Several instances plant one or more "trap" defects — softer bumps, slope strictly below `Lc`
(unlike the sharper generic ones) — each seeded near a free pilot reading, but positioned to
stay unreachable by several plausible fixed, regularly-spaced probe layouts, however such a
layout is chosen in advance. So the pilot batch already shows a genuine, nonzero, but
INCOMPLETE reading at every pilot point that happens to seed a trap (every OTHER pilot
reading, and every point no plausible fixed layout reaches, stays exactly zero). A policy that
pre-plans its probes, however evenly spaced, lays a strong baseline but never reacts to any of
these clues — it earns nothing more from them than the pilot already gave. A policy that keeps
asking whether reacting beats extending coverage spends the budget closing in on them instead.

## Constraints
`S = 12`, budget = 32 probes spent one at a time over 32 rounds (a blind, pre-planned layout
only needs 16 — the rest reward reacting to a reading), `Lc` roughly `[1.6, 3.2]`,
`m_cap = 11.0`, `n_inspect = 101`. Memory 512MB; per-call timeout is generous.
