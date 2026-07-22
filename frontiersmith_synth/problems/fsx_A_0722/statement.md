# Ladder Filter Synthesis from a Target Curve

## Problem

You are given a fixed low-pass ladder topology with **N** reactive component slots,
numbered 1..N from the source end. Odd slots (1, 3, 5, ...) are **series inductors**;
even slots (2, 4, 6, ...) are **shunt capacitors** (to ground). The source and the load
each have the same resistance **Z0**.

For each slot you must choose one component value from a fixed discrete candidate list
(one list for inductors, one for capacitors, both given in the input, both ascending).
**Index 0 in either list is always the value 0** — for an inductor that means a plain
wire (short); for a capacitor that means an open circuit (the shunt branch is simply
absent). Choosing index 0 therefore "un-populates" that slot.

You are also given a target magnitude response — a list of **M** frequencies (Hz) and
the desired |H(f)| in decibels at each — and a **cost per populated (non-zero) slot**.
Your job is to choose component values whose *actual* cascaded response matches the
target curve as closely as possible, as cheaply as possible.

**Transfer function.** For a frequency f, let ω = 2πf. A series component of impedance
Z contributes the 2×2 chain (ABCD) matrix [[1, Z],[0, 1]]; a shunt component of
admittance Y contributes [[1, 0],[Y, 1]]. An inductor of value L has Z = jωL; a
capacitor of value C has Y = jωC (value 0 gives Z = 0 or Y = 0 respectively). Cascade
the N matrices in slot order (slot 1 nearest the source) to get an overall
[[A, B],[C, D]]. Because the source and load resistances are both Z0, the voltage
transfer function is

  H(f) = Z0 / (A·Z0 + B + C·Z0² + D·Z0),  scored as |H(f)| in dB = 20·log10(|H(f)|).

Every component in the network interacts through this single cascade — changing any
one slot's value shifts the whole curve, not just "its" frequency region.

## Input (stdin)

```
N  Z0  M  cost_per_component
K_L  L_0 L_1 ... L_{K_L-1}      (ascending, L_0 = 0)
K_C  C_0 C_1 ... C_{K_C-1}      (ascending, C_0 = 0)
f_1 f_2 ... f_M                  (Hz, ascending)
target_dB_1 ... target_dB_M
```

## Output (stdout)

Exactly N whitespace-separated integers idx_1 .. idx_N: idx_i is an index into the L
list (if i is odd) or the C list (if i is even), giving the value of slot i.

## Feasibility

Output must contain exactly N tokens, each parsing as a finite integer, with
0 ≤ idx_i < K_L for odd i and 0 ≤ idx_i < K_C for even i. Any violation scores 0.

## Objective (minimize)

For each frequency, let d_i = min(|your |H(f_i)| in dB − target_dB_i|, 12.0) — the
per-point dB error, capped at 12 dB so one badly-off point deep in the stopband cannot
by itself swamp the whole score. Then

F = mean_i (d_i²) + cost_per_component × (number of populated, i.e. non-zero-index,
slots).

The target curve is not guaranteed to be exactly achievable by any component choice —
treat it as a real-world spec with a bounded tolerance, not a puzzle with a perfect
answer.

The checker also builds its own baseline B = F of the fully-unpopulated network (every
slot at index 0, i.e. the bare Z0/(Z0+Z0) resistor divider — no rolloff at all). Your
printed score is Ratio = min(1.0, 0.1 · B / F): matching the baseline scores ~0.1;
a substantially better (lower-F) fit scores higher, up to a cap of 1.0.

## Example (worked, illustrative only — not a real test instance)

N=1 (a single series inductor, slot 1), Z0=50, one frequency f=5000 Hz, target_dB =
−6.430, L-list = [0, 0.001], cost_per_component=0.05.

Choosing idx_1=1 (L=0.001 H): ω=2π·5000≈31415.9, so B=31415.9j·0.001=31.416j.
A·Z0+B+C·Z0²+D·Z0 = 50+31.416j+0+50 = 100+31.416j, |·|≈104.82,
|H|=50/104.82≈0.4770, dB≈−6.430 — matches the target, so err≈0 and
F = 0 + 0.05·1 = 0.05.

Baseline (idx_1=0, L=0): denom = 100 (real), |H|=0.5, dB≈−6.021,
err=(−6.021−(−6.430))²≈0.167, B = 0.167 + 0 = 0.167 (no components populated).

Ratio = min(1.0, 0.1 · 0.167 / 0.05) ≈ 0.335.

## Constraints

3 ≤ N ≤ 12, 20 ≤ Z0 ≤ 200, M = 30, all frequencies and target dB values finite,
cost_per_component > 0. Time limit 5s, memory 512MB.
