# Airlock Splice: Recovering a Scrambled Counter Relay

## Problem
The derelict ship's airlock control bus was severed and badly spliced
back together. Every relay pulse that crosses the bus is now logged as
one of four raw two-bit codes: `A`, `B`, `C`, `D`. Before the splice,
these four codes drove a simple pressure-differential relay:

- one code was the **PRESSURIZE** pulse: it nudges a hidden drift counter
  *up* by a fixed positive integer unit `u`;
- one code was the **VENT** pulse: it nudges the counter *down* by the
  *same* unit `u`;
- one code was the **POLARITY** chatter: every time it appears it
  *inverts the sign* of every nudge that comes after it (so a later
  PRESSURIZE pulse may actually subtract, and a later VENT pulse may
  actually add), and this inversion persists, layer upon layer, for the
  rest of the burst (a second POLARITY pulse inverts it back, a third
  inverts it again, and so on);
- one code is **NULL** keep-alive chatter with no effect at all.

The splice scrambled *which raw code plays which of the four roles*, and
it also erased the nudge unit `u` and whether the relay started in
normal or inverted polarity. All of this — the role assignment, `u`, the
starting polarity — is fixed for a given derelict ship but otherwise
completely unknown to you.

**Illustrative FORM only — not the hidden law of this problem:** e.g. a
rule where a code `R` *resets* a running total to zero every time it
appears, while two other codes add/subtract a shared unit, is the *shape*
of "one code with special persistent-ish behavior, two codes that
add/subtract." The real relay here follows the pressurize/vent/polarity/
null mechanics above, with an inversion that persists and compounds
(re-inverting on every polarity pulse), not a reset — a different
mechanism entirely.

The ship's black box logged some short chatter bursts together with the
final drift reading the relay reported after each burst. You must recover
a plausible relay wiring (role assignment + unit + starting polarity)
that explains those logs, because engineers will soon need the predicted
drift on much longer chatter bursts pulled from deep storage — far longer
than anything in your log — and only a wiring that captures the *true,
order-sensitive* mechanics will predict those correctly.

## Input (stdin)
```
t K
burst_1 y_1
burst_2 y_2
...
burst_K y_K
```
`t` is the test id. `K` bursts were logged, each `burst_i` a string over
`{A,B,C,D}` of length 8-20, and `y_i` the integer drift reading the relay
reported after that burst (occasionally, roughly one logged reading in
six, a small integer transcription error of up to +/-3 was added before
logging).

## Output (stdout)
Six whitespace-separated tokens on one line:
```
inc dec flip noop u p0
```
`inc`, `dec`, `flip`, `noop` are your recovered role assignment (a
permutation of `A B C D` naming which code is PRESSURIZE, VENT, POLARITY,
NULL respectively); `u` is your recovered positive integer nudge unit;
`p0` is your recovered starting polarity (`1` or `-1`).

## Feasibility
The submission scores `Ratio: 0.0` unless ALL hold:
- exactly 6 tokens; the first four are a permutation of `A`,`B`,`C`,`D`;
- `1 <= u <= 1000` (a plain integer);
- `p0` is exactly `1` or `-1`.

## Objective (minimize)
The grader draws several **fresh**, much longer chatter bursts (never
shown to you; lengths from about 100 up to 500) and computes the final
drift reading TWICE: once by simulating the true hidden wiring, once by
simulating your submitted wiring. Let `F` be the mean absolute difference
between the two readings over all held-out bursts.

## Scoring
```
B = mean absolute error of the "unscrambled, unit-magnitude" guess
    (inc=A, dec=B, flip=C, noop=D, u=1, p0=1)
eps = B / 8
sc = min(1000, 100 * (B + eps) / max(1e-9, F + eps))
Ratio = sc / 1000
```
Submitting that flat guess reproduces `B` exactly, scoring `Ratio = 0.1`.
Because of the `eps` floor, even an exact wiring recovery (`F = 0`) caps
at `Ratio = 0.9` — there is always headroom above any single strategy.

## Constraints
`1 <= t <= 10`; `74 <= K <= 110`; each burst has length in `[8,20]`; time
limit 5s, memory 512m. Each `.in` file is well under 5 MB.

## Example (worked score)
Suppose `B = 900` and a submission achieves `F = 150` (`eps = 112.5`):
`sc = 100*(900+112.5)/(150+112.5) = 100*1012.5/262.5 ~= 385.7`, so
`Ratio ~= 0.3857`. A submission with `F = 0` gets
`sc = 100*1012.5/112.5 = 900`, i.e. `Ratio = 0.9`.
