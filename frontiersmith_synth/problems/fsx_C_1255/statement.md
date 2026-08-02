# Spending Your Bits Where the Signal Is

## Problem
You are configuring the analog front end of a data-acquisition channel: an
ADC with tunable **resolution** `B` bits and **sample rate** `R` Hz, fed
through an antialiasing low-pass filter of tunable **order** `K`, followed
by a programmable-gain amplifier with **gain code** `G`. You are given the
channel's power spectrum as `NBINS` discrete tones `(f_i, p_i)` (frequency
in Hz, power in arbitrary units) and must recover as much power as possible
in a **target band** `[FLO, FHI]` relative to the noise that lands there.

**Resolution vs. rate.** A fixed conversion-throughput budget links `B` and
`R`: raising resolution linearly eats into the sample rate you can afford.
**Filter order.** Steeper filters (`K` larger) cost more too, drawn from the
*same* budget. **Gain.** Raising `G` shrinks the amplifier's full-scale
window `FS = PFS / 2^G` (headroom relative to a fixed device rating `PFS`),
which reduces quantization noise but risks clipping loud tones.

The antialiasing filter's cutoff is pinned to the ADC's own Nyquist
frequency `fc = R/2` (order-`K` Butterworth magnitude response). A tone at
frequency `f` survives the filter at gain
`atten(f) = 1 / (1 + (2f/R)^(2K))` (and `atten(f) = 1` when `K = 0`, i.e. no
filter). After surviving the filter, a tone with `p*atten(f)` exceeding the
full-scale window `FS` **clips**: only `FS` of it is usable, and the excess
`p*atten(f) - FS` becomes broadband **clipping noise**.

For each tone, let `usable(f) = min(p*atten(f), FS)` and
`clip(f) = max(0, p*atten(f) - FS)`. Then:
```
signal      = sum of usable(f_i) over tones with FLO <= f_i <= FHI
aliased     = sum of usable(f_i) over tones with f_i > FHI or f_i < FLO,
              but ONLY those with f_i > R/2 (beyond Nyquist -- anything
              at or below Nyquist keeps its own frequency and cannot
              fold into your band)
clip_noise  = sum of clip(f_i) over ALL tones
quant_noise = FS / 4^B
noise       = quant_noise + aliased + clip_noise + NFLOOR
```
`NFLOOR` (given) is a fixed hardware noise floor -- extra resolution or
gain beyond what's needed to push the *other* noise terms under `NFLOOR` is
wasted budget. Objective (maximize): `SNR = signal / noise`.

## Input (stdin)
```
NBINS BUDGET WFILT PFS NFLOOR FLO FHI
f_1 p_1
...
f_NBINS p_NBINS
```
`4 <= NBINS <= 40`, all values positive integers, `f_i` distinct,
`FLO < FHI`. `B in [1,16]`, `K in [0,8]`, `G in [0,12]` are FIXED global
bounds (not given per instance).

## Output (stdout) — the artifact
Four integers on one line: `B R K G`. Feasible iff `1<=B<=16`, `R>=1`,
`0<=K<=8`, `0<=G<=12`, all integers, and `B*R + WFILT*K <= BUDGET`. Any
violation, malformed token, wrong token count, or non-finite value scores
`Ratio: 0.0`.

## Scoring
The checker recomputes `SNR` exactly with exact rational arithmetic (no
tolerance) and compares it against its own baseline `B_ref`: the SNR of a
fixed reference choice `B=6, K=2, G=0`, spending the rest of the budget on
`R`. For this maximization objective:
```
Ratio = min(1.0, SNR / (10 * B_ref))
```

## Why this is a real trade, not a single knob
Pushing `B` to the max looks like the free win (quantization noise falls
exponentially in `B`), but every bit spent on resolution is a bit *not*
spent on rate, and rate sets the filter's own cutoff `fc = R/2`. Starve `R`
and `fc` drops below your own target band -- the filter that was supposed
to protect you from aliasing now attenuates the very tones you wanted to
keep, while unfiltered energy above the (now low) cutoff keeps landing in
your band. The fix is not "more filter order" alone either: order costs the
same shared budget as rate. The right allocation depends on where the
input's actual energy sits (in-band vs. the interferers above it) and how
close the true peak sits to `PFS` (unused headroom is free SNR via `G`;
using it up when there is none causes clipping).

## Example
`NBINS=2, BUDGET=100, WFILT=5, PFS=1000, NFLOOR=1, FLO=50, FHI=60`, tones
`(55,80) (200,80)`. Output `B=2 R=25 K=0 G=0`: cost `= 2*25+0=50<=100` OK.
`fc=12.5`, `atten=1` (K=0) for both tones. Tone `(55,80)`: in-band,
`FS=1000`, `usable=80`. Tone `(200,80)`: out-of-band, `f=200>fc=12.5`, so
it aliases: `usable=80`. `quant=1000/16=62.5`. `noise=62.5+80+0+1=143.5`.
`SNR=80/143.5=0.5575`.

Time limit 5s, memory 512MB.
