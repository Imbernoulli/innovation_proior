# Precursor or Perish: A Two-Rate Sampling Policy for a Year-Long Battery

## Problem

A sensor node must run for `T` discrete time slots on one battery charge of `E` energy units,
and it must be *reprogrammed once, in advance* with a single firmware policy that will then be
deployed, unchanged, across `K` independent field placements ("streams") -- each stream is a
fresh horizon of the same length `T` with its own fresh battery of `E` units, but a different
realization of what actually happens.

Two ways to spend energy in any slot:
- A **full sample** costs `e_full`. It detects an event if and only if it lands exactly on a
  slot where an event is truly present.
- A **precursor check** costs `e_cheap` (much less than `e_full`). It never detects an event by
  itself, but reads a binary flag that is `1` for exactly the `lead` slots immediately before
  every event **cluster** (a burst of consecutive event slots, length at most `Lmax`, at most
  `Cmax` clusters per stream). Isolated background events carry no such warning.

Your policy is four non-negative integers `(P0, Pc, P1, W)`:
- By default, take a full sample every `P0` slots.
- Every `Pc` slots (while not currently escalated; `Pc = 0` means never check), spend `e_cheap`
  on a precursor check.
- If a check reads `1`, **escalate**: for the next `W` slots, take full samples every `P1`
  slots instead of `P0`.

Energy is a hard per-stream cap: if a channel access would exceed the *remaining* budget for
that stream, it is simply skipped (no crash, no penalty beyond missing that access) -- so every
`(P0,Pc,P1,W)` is always "safe" to submit, but a policy that overspends early goes dark later.

## Input (stdin)

```
T E e_full e_cheap lead Lmax Cmax K
n_1
ev_1_1 ev_1_2 ... ev_1_{n_1}
m_1
pr_1_1 pr_1_2 ... pr_1_{m_1}
...                                  (one (n_k, events, m_k, precursors) block per stream, K blocks)
```
`ev_k_*` are the sorted slots (in `[0,T)`) where stream `k` truly has an event; `pr_k_*` are the
sorted slots where its precursor flag reads `1`. All values are non-negative integers.

## Output (stdout) — the artifact

```
P0 Pc P1 W
```
Exactly four integers on one line.

## Feasibility

Rejected (`Ratio: 0.0`) if: the output does not contain exactly four tokens; any token is not a
plain (optionally signed) integer (garbage, decimals, `nan`, `inf` all rejected); or the values
violate `1 <= P0 <= 10^6`, `0 <= Pc <= 10^6`, `1 <= P1 <= 10^6`, `0 <= W <= 10^6`.

## Scoring

The checker replays your policy, independently, on every one of the `K` streams (fresh `E`
each), using the rule above, and sums the total events detected across all streams into `F`.
It also computes `B`: what its own reference construction -- one fixed rate spending the
**whole** of `E`, never using the precursor channel -- would detect. Score:

```
ratio = min(1.0, 0.1 * F / max(1e-9, B))
```

More detected events (relative to that half-budget fixed-rate reference) -> higher score.

## Example

Toy instance, illustrating the MECHANISM only (not a worked score): `lead=3`, one stream has a
cluster of 4 events at slots `10,11,12,13`, preceded by precursor flag `1` at slots `7,8,9`. A
fixed rate of period 5 starting at slot 0 samples `0,5,10,15,...` -- it catches slot `10` only,
1 of 4 cluster events. A policy with `Pc=3` is guaranteed (period `<= lead`) to check at least
once inside `[7,9]`, sees the flag, and escalates with `P1=1` for `W` slots -- catching all 4.

## Constraints

`1 <= K <= 10`, `1 <= T <= 1700`, `1 <= e_cheap < e_full <= 20`, `0 <= lead <= 14`,
`0 <= Lmax <= 12`, `0 <= Cmax <= 3`, `E >= 4*e_full`. Time limit 3s, memory 512MB.
