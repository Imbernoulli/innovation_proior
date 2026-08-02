# Locating a Crack From How the Bridge Rings

A slender 1-D structure of length `L` (a "bridge deck" model) has mode shapes
`phi_m(x) = cos(m*pi*x/L)` for mode numbers `m = 1,2,3,...`, `0 <= x <= L`.
Mode `m` has `m` interior **node lines** — points where `phi_m(x) = 0` — at
`x = (2j-1)*L/(2m)` for `j = 1..m`. At its own node, mode `m` is
**perfectly insensitive** to local damage there.

A single hidden crack sits at unknown location `x*` in `(0,L)` with unknown
severity `s*` (a small fractional stiffness loss). It perturbs the structure
in two measured ways, for every mode `m` you are given:

- **Frequency**: `f_m = f0_m * (1 - s* * phi_m(x*)^2) + noise` — the
  fractional frequency drop is proportional to the squared mode-shape
  amplitude AT the crack. Exactly zero if `x*` is on mode `m`'s node line.
- **Shape**: at each of `G` coarse sensor ("gauge") positions `x_g`, the
  damaged mode-shape amplitude is
  `psi_m(x_g) = phi_m(x_g) * (1 - s* * exp(-((x_g - x*)/w)^2)) + noise`,
  `w = 0.10*L` — a localized dent in the mode shape around the crack.

You are given data for `K` measured modes only; both channels for other
modes (and the crack itself) are hidden.

## Input (stdin)

```
t L G K
m_1 m_2 ... m_K
f0_1 f0_2 ... f0_K
f_1  f_2  ... f_K
xg_1 xg_2 ... xg_G
<K lines: undamaged shape row i>   phi_{m_i}(xg_1) ... phi_{m_i}(xg_G)
<K lines: damaged   shape row i>   psi_{m_i}(xg_1) ... psi_{m_i}(xg_G)
```
`t` is the test id (informational). `10 <= L <= 80`, `3 <= K <= 6`,
`7 <= G <= 13`, mode numbers satisfy `2 <= m_i <= 14` (strictly increasing).
All frequencies/positions/amplitudes are floats.

## Output (stdout)

One line: `x_hat s_hat` — your estimate of the crack's location and
severity.

## Feasibility

`x_hat` and `s_hat` must parse as finite numbers with `0 <= x_hat <= L` and
`0 <= s_hat <= 0.5`. Any violation scores `0`.

## Objective & Scoring

Let `Loc = exp(-|x_hat - x*| / (0.05*L))`, `Sev = exp(-|s_hat - s*| / 0.07)`.
The checker also silently evaluates your estimate against **3 further modes
you were never given data for** (deterministically chosen, different node
lines from your `K` modes), using the SAME forward relation: for each such
mode `m`, `pred = s_hat * cos(m*pi*x_hat/L)^2` vs the true
`true = s* * cos(m*pi*x*/L)^2`, giving `Hold_m = exp(-|pred-true|/0.05)`
averaged into `Hold`. A location/severity pair that is only locally
consistent with the modes it was fit to, but does not generalize, scores
poorly here. The raw quality is

```
F = 0.35*Loc + 0.15*Sev + 0.50*Hold
```

The checker also builds an internal baseline `B`: guess the midpoint
`x_hat = L/2` (ignore location entirely) and `s_hat` = the plain average of
`|1 - f_m/f0_m|` over your `K` given modes, then computes the SAME `F`
formula for that guess. Your score is

```
Ratio = min(1.0, 0.1 * F / B)
```

so the naive midpoint-and-average baseline scores `~0.1`.

## What makes it hard

Fitting a single mode's frequency shift alone never pins down `(x*, s*)`
uniquely — and if that one mode happens to be blind at `x*` (its node
line), you learn essentially nothing from it, in EITHER channel: near a
node the shape amplitude itself is also near zero, so the "damage dent" is
invisible in that mode's shape data too. Different modes have different
node lines, so a location that blinds one mode is generically visible to
others. The way to be robust is to combine several modes' frequency data
at once (a mode near its own node should count for little, automatically)
and use the shape channel to break ties between locations that fit the
frequency data almost equally well (frequencies alone are periodic in `m`,
so more than one candidate can look good).

## Example (worked, illustrative form only — not from the real generator)

For a toy `L=2`, if `x*=1` (the midpoint) and `s*=0.2` with mode `m=1`:
`phi_1(1) = cos(pi/2) = 0`, so `f_1` shows **zero** shift even though real
damage is present there — mode 1 is exactly blind at the midpoint. A method
using mode 1 alone would report "no damage."

## Constraints

Time limit 5s, memory 512MB. Scoring is fully deterministic given the
input and your output.
