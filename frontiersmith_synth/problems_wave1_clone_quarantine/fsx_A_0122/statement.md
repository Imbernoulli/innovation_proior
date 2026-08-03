# Ridgeline Turnpike: Bounded-Space Gantry Staffing

## Story

A mountain turnpike clears traffic through a bank of electronic toll gantries.
Vehicles arrive already grouped into indivisible **platoons** (convoys that travel
bumper-to-bumper and must be waved through the *same* gantry in a single green
cycle). Each staffed gantry can process a total **load** of `C` per cycle — the sum
of the platoon loads routed to it (think combined axle count). A platoon is routed
**whole**; it may never be split across two gantries.

Powering and staffing a gantry is expensive, and the control cabin can keep only
`K` gantries **open at the same moment**. A gantry is *open* from the cycle it first
accepts a platoon until the cycle it accepts its last platoon; once closed it goes
dark and cannot be reopened. At no instant may more than `K` gantries be open
simultaneously.

Platoons arrive in a fixed stream (the array index is the arrival cycle). Clear the
entire stream while **opening as few gantries as possible**.

This is **bounded-space online 1-D bin packing**: platoons are items with integer
sizes, `C` is the bin capacity, `K` caps how many bins may be open at once, and the
number of gantries opened is the number of bins used, which you MINIMIZE. The
bounded-space cap is what makes this genuinely *online* — you cannot lay down an
unconstrained offline packing, because scattering one gantry's platoons across the
arrival stream would force too many gantries open at the same time.

## You write a program (isolated stdin → stdout)

Your program reads ONE JSON object (the public instance) from stdin and writes ONE
JSON object (your routing) to stdout. It runs in a fresh, isolated subprocess and
only ever sees the public instance below.

### Input (stdin) — the public instance
```json
{
  "name": "turnpike4101",
  "capacity": 20,          // C: max total load per gantry per cycle (int)
  "n": 24,                 // N: number of platoons
  "max_open": 3,           // K: max gantries open simultaneously (int)
  "platoons": [7, 3, 18, ...]   // N integer loads, 1 <= s_i <= C, in ARRIVAL order
}
```

### Output (stdout) — your routing
```json
{ "assign": [g_0, g_1, ..., g_{N-1}] }
```
`g_i >= 0` is the gantry index platoon `i` is routed to. Gantry indices need not be
contiguous; a gantry "exists" iff at least one platoon uses it, and the number of
DISTINCT gantries is what gets minimized.

## Feasibility

A routing is **valid** iff ALL of the following hold:

1. `assign` is a list of exactly `N` non-negative integers.
2. **Capacity:** no gantry's total routed load exceeds `C`.
3. **Bounded space:** for gantry `g`, let `first(g)` / `last(g)` be the smallest /
   largest arrival index routed to it; `g` is *open* over cycles
   `[first(g), last(g)]`. For every cycle `i in 0..N-1`, the number of gantries `g`
   with `first(g) <= i <= last(g)` must be `<= K`.

Invalid output, wrong length, an overfilled gantry, a bounded-space violation, a
crash, a timeout, or non-JSON → that instance scores **0.0**.

## Scoring (deterministic, no wall-time)

For each instance the evaluator computes:

- `q_lb`   = `ceil(sum(platoons) / C)` — the L1 lower bound (generally unreachable),
- `q_base` = gantries opened by an internal **next-fit** operator (weak `K=1` baseline),
- `q_cand` = gantries opened by *your* routing.

Your per-instance score is the affine anchor
```
r = clip( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```
Matching next-fit scores ≈ `0.1`; reaching the L1 ideal scores `1.0`; doing worse
than next-fit scores `< 0.1`. The overall **Ratio** is the mean of `r` over a fixed,
seeded family of 12 instances (with harder, larger held-out cases). Because L1 is a
loose bound and only `K` gantries may be open at once, even strong bounded-space
packers stay well below `1.0` — there is real headroom.

## Notes / strategy hints

- Next-fit uses a single open gantry and reproduces the baseline (≈ 0.1).
- Keeping up to `K` gantries open (bounded first-fit / best-fit) reuses room that
  next-fit throws away. The interesting decisions are *which* open gantry to route a
  platoon to and *which* gantry to close when you must open a new one under the
  `K`-open cap.
- The candidate is untrusted and isolated: the answer key and references live only
  in the evaluator process, so introspection buys nothing.
