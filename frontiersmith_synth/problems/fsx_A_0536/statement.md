# Resistance Portrait: Fitting a Sensor Mesh to Prescribed Electrical Distances

You are wiring a mesh of `n` sensor terminals. Each wire you install between two
terminals is a resistor described by its **conductance** `c > 0` (higher conductance =
lower resistance). When the mesh is powered, the natural "electrical distance" between
two terminals `i` and `j` is their **effective resistance** `R(i,j)` — the voltage that
appears across `i`,`j` when one unit of current is pushed in at `i` and drawn out at `j`.
Formally, with Laplacian `L` (`L = D − A`, `A` the conductance-weighted adjacency),
`R(i,j) = (e_i − e_j)^T L^+ (e_i − e_j)`, where `L^+` is the Moore–Penrose pseudoinverse.

A calibration spec gives you `P` **target pairs**; pair `k` names terminals `i_k`,`j_k`,
a desired effective resistance `t_k > 0`, and an importance weight `w_k > 0`. Build a
mesh whose realized effective resistances match the targets as closely as possible.

**The catch (read this).** Effective resistances are *not* independent knobs. By Rayleigh
monotonicity, adding or strengthening *any* wire can only **lower** every pairwise
effective resistance. So a wire you install to fix one pair also pulls down every other
pair, and the whole portrait moves together through the shared pseudoinverse. Setting a
pair's direct wire to `1/t_k` overshoots (the realized resistance comes out *below* `t_k`)
whenever that pair shares terminals or a backbone with others. Good designs pick a small
shared skeleton and reason about the *joint* response, not one pair at a time.

## Input (stdin)
```
n m wmax P
i_1 j_1 t_1 w_1
...
i_P j_P t_P w_P
```
`n` terminals `0..n−1`; edge budget `m` (max number of wires); conductance cap `wmax`;
`P` target pairs. `m < P` in general — you cannot afford a private wire per pair.

## Output (stdout)
```
E
u_1 v_1 c_1
...
u_E v_E c_E
```
`E ≤ m` undirected wires. Each wire needs `0 ≤ u,v < n`, `u ≠ v`, `0 < c ≤ wmax`, and no
repeated unordered pair `{u,v}`. **Every specified target pair must be connected** by the
mesh (otherwise its resistance is infinite and the submission scores 0).

## Objective (minimize)
Let `R(i_k,j_k)` be the realized effective resistances. The error is the weighted RMS
**relative** error
```
F = sqrt( ( Σ_k w_k · ((R(i_k,j_k) − t_k)/t_k)^2 ) / ( Σ_k w_k ) ).
```

## Scoring
The checker builds a baseline mesh `B` (a uniform conductance-1 backbone path over the
active terminals) with error `F_B`, then reports
```
Ratio = min(1.0, 0.1 · F_B / F).
```
Matching the baseline scores ≈ 0.1; an error ten times smaller caps at 1.0. Targets are
jointly infeasible under the budget and cap, so the score has real headroom.

## Constraints
`n ≤ 48`, `P ≤ ~160`, `wmax` given per instance (small — a hard conductance cap that keeps
some targets out of reach), time limit 5s, memory 512 MB.

## Example
`n=3, m=2, wmax=25, P=1`, target pair `0 2 1.0 1.0`. Output `2 / 0 1 2.0 / 1 2 2.0`
puts two conductance-2 wires in series between 0 and 2: `R(0,2)=1/2+1/2=1.0`, hitting the
target exactly (`F=0`, Ratio 1.0). A single direct wire `0 2 1.0` would also work here —
but with more, overlapping targets no such per-pair trick exists.
