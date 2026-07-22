# Null-Space Steering: A Redundant Arm on a Drawn Path

## Problem
A planar robot arm is anchored at the origin and built from `N` rigid links
of fixed lengths `L_1, ..., L_N` (`N >= 5`), connected by revolute joints.
Joint `i` has angle `theta_i` measured relative to the direction of link
`i-1` (link `0` is the positive x-axis). Writing the cumulative direction
`phi_i = theta_1 + ... + theta_i`, the joint positions are
`p_0 = (0,0)` and `p_i = p_{i-1} + L_i * (cos(phi_i), sin(phi_i))`. The
end-effector is `p_N`.

Because `N >= 5` and the end-effector position is only 2 numbers, the arm is
**redundant**: infinitely many joint-angle vectors reach the same point.
You are given an ordered sequence of `M` end-effector waypoints (a drawn
path) and `K` circular obstacles. For **each** waypoint output a full
joint-angle vector placing the end-effector exactly there. Since consecutive
waypoints can be satisfied by very different configurations, the redundant
freedom must be spent wisely and **jointly across the whole path**, so that
(a) no link ever crosses an obstacle at any waypoint and (b) the
configuration changes little between consecutive waypoints (a physical arm
cannot teleport its joints).

## Input (stdin)
```
N M K
L_1 L_2 ... L_N
x_1 y_1          (waypoint 1)
...
x_M y_M          (waypoint M)
cx_1 cy_1 r_1    (obstacle 1)
...
cx_K cy_K r_K    (obstacle K)
```
All values are reals. Waypoints are listed in the order they must be
visited; obstacles are circles the arm must never intersect.

## Output (stdout)
`M` lines, each with `N` reals `theta_1 ... theta_N`: the joint-angle vector
(radians) at the corresponding waypoint, in the same order as the input.

## Feasibility
An output is valid iff **all** hold:
- exactly `M*N` finite real numbers are printed (no `nan`/`inf`);
- every `theta_i` lies in `[-pi, pi]`;
- for every waypoint `i`, the forward-kinematics end-effector position lies
  within `0.03` (Euclidean) of `(x_i, y_i)`;
- for every waypoint `i` and every link segment `(p_{k-1}, p_k)`, the
  distance from the segment to every obstacle's center is `>= r` (no link
  ever enters an obstacle disk, at any waypoint).
Any violation scores `Ratio: 0.0`.

## Objective
Minimize the total joint-space travel along the path:
```
F = sum_{i=1}^{M-1} sum_{j=1}^{N} | wrap(theta_j^{(i+1)} - theta_j^{(i)}) |
```
where `wrap(.)` reduces an angle difference to `(-pi, pi]` (shortest turn per
joint). `F` is exactly the total amount every joint has to rotate to sweep
the arm through the whole path -- a proxy for how physically smooth and
achievable the motion is.

## Scoring
The checker builds its own reference path `B`: for each waypoint
*independently*, always starting from the arm's neutral rest pose (all
joints at `0`), it finds *some* configuration reaching the waypoint without
colliding (memory-less but obstacle-aware; it never relates one waypoint's
configuration to the next). Given your total travel `F` and the baseline's
total travel `B`:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching the baseline's travel scores `Ratio = 0.1`; achieving `10x` less
total joint travel than the baseline caps the score at `1.0`. Since the
baseline resets every waypoint, it typically wastes a lot of motion -- a
solver that keeps the arm's shape continuous from one waypoint to the next,
*and* only spends its spare joints to dodge obstacles when actually needed,
can travel far less than `B`.

## Constraints
- `5 <= N <= 8`, `10 <= M <= 18`, `0 <= K <= 3`.
- `0.7 <= L_i <= 1.3`. All waypoints lie strictly inside the arm's reachable
  disk and strictly outside every obstacle; obstacles never cover the
  origin.
- Time limit 5s, memory 512m.

## Example
Suppose `N=5`, all `L_i=1`, and the path only has `M=2` waypoints that both
happen to be reachable with the *same* joint-angle vector `theta`. Then
`F = 0` (zero travel) -- but the checker's baseline `B` almost never
reaches `0` (its two independent solves generally land on different
branches), so `sc = min(1000, 100*B/max(1e-9,0)) = 1000`, `Ratio = 1.0`.
More realistically, an output whose total travel `F` equals the baseline's
`B` scores `Ratio = 0.1`; halving it scores `Ratio = 0.2`.
