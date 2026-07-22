# Clinic Roster: Staggering the Boundary Against the Surge Lattice

## Problem

You staff a walk-in clinic that runs every day on the same 24-hour rhythm. You
have exactly **W** workers and a fixed shift **rule**: every worker works one
shift of exactly **L** consecutive hours per day (wrapping past midnight if
needed), and a shift may only *start* at an hour `h` (0..23) satisfying
`h mod g == r` (a published start-hour grid). Each worker on duty handles
**C** patients per hour. You must decide, for each of the W workers, which
allowed hour their daily shift starts at — the same roster repeats every day.

Patients arrive according to a published base hourly curve `base[0..23]`
(patients arriving during hour-of-day `h`, every day). On top of the base
curve, the clinic must survive a published **sweep** of **K** possible surge
profiles — a rush the clinic might face is only known to be *one of* these
K published shapes, never in advance. Surge profile `k` is a triple
`(s_k, d_k, a_k)`: during hours `[s_k, s_k+d_k)` of *every* day (wrapping mod
24), `a_k` extra patients arrive on top of the base curve.

For a fixed roster, coverage at hour-of-day `h` is `cap[h] = C * (number of
workers whose shift covers hour h)`. For a single surge profile `k`, simulate
a deterministic queue over 3 days (72 hours, `t = 0..71`), starting empty:

```
Q[0] = 0
Q[t+1] = max(0, Q[t] + arrivals[t] - cap[t mod 24])
arrivals[t] = base[t mod 24] + (a_k if (t mod 24 - s_k) mod 24 < d_k else 0)
```

The first 24 hours are a warm-up; record `peak_k = max(Q[t] for t in
24..72)`. Your objective is the **worst peak over the whole sweep**:
`F = max_k peak_k` for k = 1..K. **Minimize F.**

Illustrative FORM only (not the mechanism above): if a checker scored
"minimize the sum of squared deviations from a target vector," worse
solutions would simply have larger deviations everywhere — that is *not*
how this objective behaves; here a single badly-placed hour can dominate F
regardless of how good every other hour is, because F is a max over the
sweep, not an average.

## Input (stdin)

```
W L C g r
base[0] base[1] ... base[23]
K
s_1 d_1 a_1
...
s_K d_K a_K
```
All values are non-negative integers; `1 <= g <= 3`, `0 <= r < g`,
`1 <= L <= 14`, `6 <= W <= 30`, `1 <= K <= 30`.

## Output (stdout)

Exactly **W** integers (whitespace/newline separated): the start hour (each
in `0..23`, each satisfying `h mod g == r`) of every worker's shift, in any
order.

## Feasibility

The output must contain exactly W tokens, each parsing as a finite integer
in `[0,23]` satisfying the start-hour rule `h mod g == r`. Any violation
(wrong count, non-integer, non-finite, out of range, or off the allowed
grid) scores `Ratio: 0.0`.

## Scoring

Let `F` be your worst-case peak queue (see above). The checker also builds
its own baseline roster `B_roster` (every worker starts at the same first
allowed hour) and computes its worst-case peak `B`. Score:

```
sc = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```

Smaller F (relative to the clustered baseline) scores higher. The score is
capped, so a solution does not need to reach F=0 to score near the top.

## Constraints

`6 <= W <= 30`, `1 <= L <= 14`, `1 <= C <= 8`, `1 <= g <= 3`, `0 <= r < g`,
`0 <= base[h] <= 400`, `1 <= K <= 30`, `0 <= s_k < 24`, `1 <= d_k <= 6`,
`1 <= a_k <= 800`. Time limit 5s, memory 256MB.

## Example (worked, small illustrative numbers — not from the real generator)

`W=2 L=12 C=2 g=1 r=0`, `base` all zero, `K=1`, profile `(0, 2, 10)`.
Roster `[0, 12]` covers every hour with 1 worker (`cap[h]=2` all day), so
`arrivals` during the surge hours is 10 vs capacity 2/hour for 2 hours:
`Q` rises to 8 then drains; `peak_1 = 8` (roughly, once warm-up settles).
A different roster, say `[6, 18]`, gives the *same* total coverage shape
but shifted in phase — if the surge profile instead started at hour 6, that
roster would take the hit and `[0,12]` would not. The real instances plant
many such surge profiles, and the roster that "spreads workers evenly
starting at hour 0" is not automatically the roster whose weak hours are
farthest from the published sweep.
