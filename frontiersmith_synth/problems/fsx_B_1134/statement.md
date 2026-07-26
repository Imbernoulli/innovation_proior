# Fold-Chain Inverse Kinematics

## Problem
A crease pattern is a chain of `K` degree-4 vertex "flowers" glued corner to corner.
Flower `k` has an interior vertex `v_k`, four creases meeting there, four sector
angles `a_k1..a_k4` (radians, `a_k1+a_k2+a_k3+a_k4 = 2*pi`, and `a_k1+a_k3 =
a_k2+a_k4 = pi` -- the Kawasaki condition, so the vertex is genuinely rigid-foldable
away from flat) and four crease lengths `L_k1..L_k4`. Its four triangular panels are
`(v_k, p_ki, p_k(i+1))`; crease `i` is the shared edge `(v_k, p_ki)`. Consecutive
flowers are joined by an extra hinge at corner `p_k3`: that point is both flower
`k`'s third corner AND flower `k+1`'s vertex `v_(k+1)` -- a fixed pivot, unaffected
by its own hinge angle, that carries the rest of the chain around with it.

Folding assigns a dihedral angle to every crease. Around a single flower, going
crease-1 → crease-2 → crease-3 → crease-4 and back to crease-1 must compose (as a
3-D rotation, in order) to the identity -- otherwise the sheet tears at that vertex.
This is the loop-closure / rigid-foldability constraint; for a Kawasaki vertex the
angles that satisfy it trace exactly a 1-parameter curve.

You output every crease angle. The checker computes, for each flower `k`, the 3-D
position its corner `p_k3` actually reaches (an isometric image of the flat sheet --
lengths and sector angles never change, only dihedral angles do) and compares it to
a given target point, while also measuring how badly loop-closure was violated.

## Input (stdin)
```
K
a_11 a_12 a_13 a_14
L_11 L_12 L_13 L_14
...  (repeated for each of the K flowers)
tx_1 ty_1 tz_1
...  (K target points, one per flower's p_k3)
```

## Output (stdout)
`4*K + (K-1)` whitespace-separated floats, each in `[-3.0, 3.0]` radians and finite:
for flower `1..K`: `theta1 theta2 theta3 theta4`, and after every flower except the
last, one extra bridge fold angle for the hinge to the next flower.

## Feasibility
Exactly `4*K+(K-1)` finite tokens, each parseable as a float with `|angle| <= 3.0`.
Any violation (wrong count, non-numeric, `nan`/`inf`, out of range) scores `0`.
Tearing (loop-closure violation) is NOT a hard-reject -- it is a graded penalty (see
Scoring): a slightly inconsistent fold still scores something, a badly torn one
scores close to nothing.

## Objective (minimize)
```
dist      = mean over the K flowers of || achieved p_k3 - target_k ||
tear_pen  = mean over the K flowers of the loop-closure residual (radians:
            the rotation angle of Rot(u1,theta1)  o  R4(theta2,theta3,theta4)
            away from the identity)
F = dist + 0.8 * tear_pen
```

## Scoring
The checker also evaluates `F` on the flat construction (every angle = 0, always
loop-closure-exact) to get baseline `B`. Score is `min(1000, 100*B/F) / 1000`,
so leaving the sheet flat scores ~0.1, and a fold worth ~10x the flat baseline caps
at 1.0. Every target is placed near (not exactly on) a genuinely reachable folded
configuration, so an exact match is impossible and headroom remains above the
reference solutions.

## Constraints
`1 <= K <= 3`, `0.55 <= a_ki <= pi-0.55`, `1.0 <= L_ki <= 2.4`. Time limit 5s.

## Example (worked, illustrative only)
`K=1`, `a = (1.0, 1.4, pi-1.0, pi-1.4)`, `L = (1,1,1,1)`, target `(0.3, 0.9, 0.5)`.
Flat (`0 0 0 0`) reaches `p_3 = (-0.737, 0.675, 0)`, giving baseline `B = dist =
1.173` (`tear_pen=0`, so `B=F_flat`).
Submission `"0 0.5 0 0"` (fold only crease 2, leave the loop-closure-dependent
creases at 0) reaches `dist=0.980` but has `tear_pen=0.500` (real inconsistency at
the vertex): `F = 0.980 + 0.8*0.500 = 1.380`, ratio `= min(1000,100*1.173/1.380)/1000
= 0.0850`.
Submission `"-0.1087 0.5 0.1087 0.5"` (crease 2 = 0.5 as before, but creases 1,3,4
solved so the vertex closes) reaches the SAME `dist=0.980` with `tear_pen ~ 0`:
`F = 0.980`, ratio `= 0.1197` -- noticeably higher, purely from respecting the
constraint, before even searching for a better `theta2`.
