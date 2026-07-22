# Reconfiguration Bench: Ordering Experiments on a Bit-Set Instrument

A beamline instrument has `M` binary control lines (cryo-valves, shutters, steering
magnet relays). Line `j` costs `c_j` instrument-seconds to switch, whether it flips
0→1 or 1→0. A few lines are massive and slow; most are solid-state and nearly free.

You must run a campaign of `N` experiments. Experiment `i` requires an exact on/off
pattern `s_i`: a string of `M` characters where `s_i[j] = '1'` means line `j` is on
during that experiment. Every experiment must be run **exactly once**. The instrument
starts with all lines off (`0...0`) and may be left in any state at the end.

Reordering the experiments is free, but every flip of a line — during the initial
setup from the all-off state, or between two consecutive experiments — costs `c_j`.
Choose the visiting order.

## Input (stdin)

```
N M
c_0 c_1 ... c_{M-1}
s_0
s_1
...
s_{N-1}
```

`c_j` are integers. Each `s_i` is a `0/1` string of length `M`; character `j`
describes line `j`. All configurations are distinct.

## Output (stdout)

`N` whitespace-separated integers: a permutation of `0 .. N-1`, the order in which
to run the experiments.

## Feasibility

The output must contain exactly `N` integer tokens, each in `[0, N)`, all distinct.
Anything else (wrong count, out of range, duplicates, non-integer tokens) scores 0.

## Objective (minimize)

Total reconfiguration cost

```
F = w(0, s_{p_0}) + sum_{k=1..N-1} w(s_{p_{k-1}}, s_{p_k})
```

where `p` is your permutation and `w(x, y) = sum_{j : x_j != y_j} c_j` is the
weighted Hamming distance (the first term is the setup cost from the all-off state).

## Scoring

The checker computes `B`, the total cost of running the experiments in input order
`(0, 1, ..., N-1)`, with the same formula. Your score is

```
ratio = min(1, 0.1 * B / F)
```

Higher is better. Reproducing the input order scores exactly 0.1; an order about ten
times cheaper than input order saturates the score.

Useful intuition: the total cost is usually dominated by **how often the expensive
lines flip**, not by how many flips happen overall. Running many experiments while an
expensive line stays put amortizes that line's flip across the whole block, and cheap
lines can toggle freely inside it.

## Constraints

- `2 <= N <= 2500`, `2 <= M <= 16`, `1 <= c_j <= 100`
- Time limit 5 s, memory 512 MB.

## Example

Input:

```
3 3
10 2 1
010
100
011
```

Input order `(0,1,2)`: setup of `010` flips line 1 (cost 2); `010 -> 100` flips
lines 0,1 (cost 12); `100 -> 011` flips all three (cost 13). Total `B = 27`.

Output `2 0 1` runs `011, 010, 100`: setup of `011` costs 3; `011 -> 010` flips
line 2 (cost 1); `010 -> 100` flips lines 0,1 (cost 12). Total `F = 16`, which is
optimal here — the expensive line 0 flips only once. Score
`min(1, 0.1 * 27 / 16) = 0.16875`.
