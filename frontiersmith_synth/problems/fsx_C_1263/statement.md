# Peak, Not Ceiling: Joint Power and Modulation Allocation on a Nonlinear Fibre Link

## Problem

A fibre carries `C` wavelength channels through `S` amplified spans. Span `s` adds
amplified-spontaneous-emission (ASE) noise `ase[s]`; the total ASE noise floor seen by
every channel is `N_ASE = ase[1] + ... + ase[S]` (more spans, i.e. longer haul, means
strictly more accumulated amplifier noise).

Each channel `c` also suffers fibre-nonlinearity noise that depends on its OWN launch
power `P_c` **and** on its neighbours' launch powers (cross-phase modulation between
wavelengths sharing the fibre):

```
Q_c(P)   = sum over c' != c of kappa[c][c'] * P_c'^2
NLI_c(P) = eta_c * P_c^3 + Q_c(P) * P_c^2
SNR_c(P) = P_c / (N_ASE + NLI_c(P))          (SNR_c = 0 if P_c = 0)
```

Raising `P_c` alone raises `SNR_c` almost linearly at low power (ASE-limited), but the
`eta_c * P_c^3` term eventually dominates and `SNR_c` **falls** as `P_c` grows further —
there is an interior power that maximises `SNR_c`, and because `Q_c` depends on every
other channel's power, that interior optimum shifts with what your neighbours are doing.

You must assign each channel `c` a launch power `P_c` in `[0, Pmax_c]` and a modulation
tier `m_c`. Tier `0` means the channel is silent (contributes nothing, and its power must
then be exactly `0`). Tiers `1..K` are given in the input as `(bps_k, req_k)` pairs
(bits/symbol, required linear SNR) with `bps` and `req` both increasing in `k`. Channel
`c` may use tier `m_c >= 1` only if its actual `SNR_c(P)` (computed from the FULL power
vector you chose, including every other channel's power) reaches `req[m_c]`.

## Input (stdin)

```
C S
ase_1 ase_2 ... ase_S
eta_1 eta_2 ... eta_C
K
bps_1 req_1
...
bps_K req_K
Pmax_1 Pmax_2 ... Pmax_C
baud
kappa row for channel 0  (C values)
kappa row for channel 1  (C values)
...
kappa row for channel C-1  (C values)
```
`kappa[c][c] = 0`. All values are non-negative; `ase`, `bps` integers, everything else a
decimal. `baud` is a fixed symbol-rate multiplier applied uniformly to every channel's
throughput (it scales the objective but not which strategy is better).

## Output (stdout)

`C` lines, one per channel in order `0..C-1`:
```
P_0 m_0
P_1 m_1
...
P_{C-1} m_{C-1}
```
`P_c` a real number, `m_c` an integer in `[0, K]`.

## Feasibility

Rejected (score 0) if: fewer than `2C` well-formed finite tokens; any `m_c` outside
`[0, K]`; any `P_c < 0` or `P_c > Pmax_c`; a tier-`0` channel with nonzero power; or, for
any channel with `m_c >= 1`, its true `SNR_c` (from the full submitted power vector)
falls short of `req[m_c]`.

## Scoring

Let `F = baud * sum(bps[m_c])` over all channels (throughput). Let `B` be the throughput
of the checker's own DELIBERATELY WEAK reference construction: for every channel, spend
just enough power to clear tier 1's threshold IGNORING cross-channel noise entirely
(self-noise only), then use tier 1 and never try for anything higher — `B = C * bps_1 *
baud`. Maximization score:
```
ratio = min(1.0, 0.1 * F / max(1e-9, B))
```

## Example

Toy instance, NOT a worked score: 1 channel (no cross term), `N_ASE = 10`, `eta = 0.001`,
`Pmax = 40`, tiers `(bps=1, req=0.9)` and `(bps=2, req=1.1)`. The SNR-maximising power is
`peak0 = (N_ASE/(2*eta))^(1/3) = 5000^(1/3) ~= 17.1`, giving `NLI = eta*peak0^3 = 5.0`
and `SNR = 17.1/15.0 ~= 1.14` — tier 2 is reachable (`2 bps`). Launching at the ceiling
`P=40` instead gives `NLI = 0.001*40^3 = 64`, `SNR = 40/74 ~= 0.54` — below even tier 1's
`req=0.9`, so this channel could only be submitted as tier 0 (silent, `0 bps`). "Use all
the power you're allowed" turns a working `2 bps` channel into a dead one; the profitable
power is set by the noise balance of the specific instance, not by the ceiling.

## Constraints

`3 <= C <= 12`, `1 <= S <= 30`, `K = 4`, `ase[s] in [3,5]`, `eta_c in [5e-7, 2e-6]`,
`Pmax_c in [1, 2000]`, `kappa[c][c']` small nonnegative reals (typically `<= 1e-6`).
Every instance is guaranteed to admit a feasible, nonzero-throughput assignment (e.g.
every channel silent scores `0` but is always feasible; some positive-power, tier-1
assignment always exists too). Time limit 5s, memory 512MB.
