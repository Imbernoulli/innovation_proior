# Beat Forge: A Slow Shimmer from Forbidden-High Strings

You must compose a **slow amplitude shimmer** using only a handful of **high-pitched
strings**. A luthier hands you a target envelope `E(t)` — a gentle, strictly positive
waveform that rises and falls slowly. Your instrument can play at most `k` sinusoids,
but **every playable pitch is confined to a high band** `[F_min, F_max]`. The target's
own slow wiggles live *below* `F_min` and are therefore **forbidden** — you cannot play
them directly.

The listener never hears the raw pitches (they are far too fast); the room only exposes
the **sliding-RMS envelope** of the total sound. Your task is to choose the pitches,
loudnesses and phases so that the room's envelope traces `E(t)` as closely as possible.

## Input (stdin)
```
N W F_min F_max k A            # ints: samples, RMS window, band, budget, #amplitudes
a_1 a_2 ... a_A                # the A allowed amplitude values (a discrete set)
E_0 E_1 ... E_{N-1}            # N samples of the target envelope, E_n > 0
```

## Output (stdout)
Between `1` and `k` oscillator triples, whitespace-separated:
```
f  a  phi
```
- `f` — an **integer** frequency with `F_min <= f <= F_max` (the forbidden-low band is
  everything below `F_min`).
- `a` — an amplitude, which must be **exactly one of the allowed values**.
- `phi` — a real phase in radians (any finite value).

## How the score is computed
The checker synthesizes the superposition on `N` samples (period `N`):
```
s[n] = sum_j a_j * cos(2*pi*f_j*n/N + phi_j)
```
extracts the envelope with a **fixed, centered, circular sliding-RMS** of window `W`
```
env[n] = sqrt( (1/W) * sum_{d=-(W-1)/2}^{(W-1)/2} s[(n+d) mod N]^2 )
```
and measures the root-mean-square error to the target
```
F = sqrt( mean_n (env[n] - E_n)^2 )   (minimize).
```
Let `B` be the error of the best **flat** envelope (a constant equal to `mean(E)`, i.e.
the intrinsic variation of `E`). The reported ratio is
```
Ratio = min(1000, 100 * B / F) / 1000
```
A do-nothing flat envelope scores `0.1`; cutting the error to `B/10` caps at `1.0`.
Any constraint violation, wrong token count, or non-finite value scores `0`.

## Feasibility
- Output has a multiple of 3 tokens, giving `1..k` oscillators.
- Every `f` is an integer inside `[F_min, F_max]`; every `a` is an allowed amplitude;
  every value is finite.

## The catch
A high pitch on its own produces a **flat** envelope — no shimmer at all. Slow motion in
the envelope can only appear as **beats**: two nearby high pitches `f` and `f+g` interfere
to make the envelope pulse at the *difference* frequency `g`. So the real degrees of
freedom are not the pitches themselves but their **pairwise spacings** `g`, which are free
to be small even though the pitches must be high. Matching `E` directly in pitch space is
impossible; the shimmer must be *manufactured* from difference frequencies. Note that the
envelope is a nonlinear (RMS) readout, the window `W` attenuates fast beats, and the budget
`k` may be too small to place a pair for every wiggle in `E` — so choose your spacings,
loudness pairings and phases with care.

## Example (illustrative, not an optimal answer)
With `k = 4` you may output two pairs sharing a slow spacing, e.g.
```
120 0.75 0.000000
124 0.75 1.570796
300 0.50 0.000000
307 0.50 3.141592
```
Here the first pair beats at spacing `4` and the second at spacing `7`, painting two slow
components into the envelope. Whether this scores well depends entirely on how those beats,
their strengths and phases line up with the target — that is the design problem.

## Constraints
`N = 1024`, `W = 41`, band width up to a few hundred, `k` up to `10`, `A = 6`.
Time limit 5 s, memory 512 MB. Deterministic scoring; identical output always scores
identically.
