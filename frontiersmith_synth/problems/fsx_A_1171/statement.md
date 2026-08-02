# Hidden Chain: Minimum-Cardinality Drift From Terminal Readings

## Problem
A resistor network has some nodes you can reach with a probe (**terminals**, numbered
`0 .. n_terminal-1`, with node `0` permanently grounded at `0V`) and some nodes you cannot
reach at all (**interior** nodes). One node, `0`, is the fixed voltage reference for every
measurement. A handful of resistors in the network have **drifted** from their datasheet
(nominal) value to an unknown new value; every other resistor is exactly at nominal.

You are given the full netlist (every resistor's two endpoints and its *nominal* resistance)
and a set of **excitation patterns** that have already been run on the *real*, possibly
drifted circuit: pattern `i` injects a current `Q_i` at terminal `s_i` and extracts it at
terminal `g_i` (every other node has zero net current), and reports the resulting voltage at
`s_i` and at `g_i` (nodal / Kirchhoff analysis, DC, exact resistor-network physics). Your job
is to report which resistors drifted and to what value.

The catch: with only one excitation pattern, many different single- or few-resistor
explanations can reproduce that one pattern's readings exactly (the interior nodes hide the
details of *how* current got from source to sink). An explanation that only fits one pattern
is not trustworthy -- it must also be consistent with every *other* pattern you were shown,
and it must generalize to excitations you were **not** shown at all. Your score rewards
explaining the network with **as few claimed drifted components as possible**.

## Input (stdin)
```
testId
n_terminal n_edges
n_edges lines: u v R_nominal        (0-indexed nodes; edge index = its line's 0-based order)
n_shown
n_shown lines: s g Q V_s V_g        (excitation pattern: source s, sink g, current Q;
                                      measured voltage at s and at g under the TRUE circuit)
```
Nodes `0 .. n_terminal-1` are terminals; the rest are interior. `R_nominal` are positive
integers. `Q > 0`. Node `0` is ground (`V=0`) for every pattern.

## Output (stdout)
```
k
k lines: edge_index new_resistance_value
```
Report the number of components you believe drifted, then each one's edge index (into the
given netlist) and your claimed new resistance (a positive real number).

## Feasibility
An output is valid iff **all** hold: `0 <= k <= n_edges`; every `edge_index` is a valid,
distinct edge of the netlist; every `new_resistance_value` is finite, positive, and at most
`1e7`. Any violation scores `Ratio: 0.0`.

## Objective
Build the **claimed circuit**: every claimed edge set to its claimed value, every other edge
at its nominal value. The checker holds a small set of *held-out* excitation patterns you
never saw (fresh source/sink terminals, run only on the TRUE circuit). For each held-out
pattern, compare the claimed circuit's predicted reading against the true one; a pattern
counts as **explained** in proportion to how close the prediction is (full credit at zero
error, decaying to no credit past a fixed relative-error tolerance). Let `E` = the average
explained-fraction across held-out patterns. Your objective is
```
F = E / max(1, k)
```
i.e. explain everything, with as few claimed components as possible.

## Scoring
Let `B` be the same quantity `E` evaluated at the **empty** claim (`k=0`, "nothing drifted"),
floored at a small positive constant so it never vanishes. With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Claiming nothing reproduces `B` exactly and scores `Ratio = 0.1`; correctly explaining the
held-out patterns with the true minimum number of components scores well above that.

## Constraints
- `1 <= testId <= 10`; instance size (nodes/edges) grows mildly with `testId`.
- Exactly 2 shown patterns per instance; a further 3 patterns (never printed to you) are
  held out for scoring.
- Time limit 5s, memory 512m per test case.

## Example
Suppose claiming zero drifted components gives `E = B = 0.30` (some held-out patterns happen
to be insensitive to the true fault). A submission claiming exactly the right single resistor
achieves `E = 0.95` with `k = 1`, so `F = 0.95`, `sc = min(1000, 100*0.95/0.30) = 316.7`,
`Ratio = 0.3167`. A submission claiming three resistors (only one of them right) with the
same `E = 0.95` scores `F = 0.95/3 = 0.3167`, `Ratio = 0.1056` -- explaining the same evidence
with more claimed faults is worth much less.
