# Half-plane intersection feasibility

## Research question

You are given `m` half-planes in the plane. Each half-plane is the closed set of points `(x, y)`
satisfying one linear inequality

```
a*x + b*y <= c
```

with integer coefficients `a`, `b`, `c` and `(a, b) != (0, 0)`. Decide whether there exists a single
point `(x, y)` (real coordinates) that lies in **all** `m` half-planes simultaneously — i.e. whether
the intersection of the half-planes is non-empty. Half-planes are **closed**, so a point on a boundary
line counts as inside.

Output `YES` if such a point exists and `NO` otherwise.

This is the planar two-variable linear feasibility (LP feasibility) problem stated geometrically. It is
the core sub-routine inside kernel/visibility computations, linear-programming relaxations, and
"is this set of constraints satisfiable" checks, so getting the empty / unbounded / degenerate corners
exactly right matters.

## Input / output contract

- Input (stdin): the first token is `m` (`0 <= m <= 2*10^5`). Then `m` lines follow, each with three
  integers `a b c` (`-10^6 <= a, b, c <= 10^6`, `(a, b) != (0, 0)`) describing the half-plane
  `a*x + b*y <= c`.
- Output (stdout): a single line, `YES` if the intersection of all `m` half-planes is non-empty,
  otherwise `NO`. With `m = 0` there are no constraints, so the answer is `YES`.
- Time limit: 2 seconds. Memory: 256 MB.

Example 1:

```
3
0 -1 0
-1 1 1
1 1 4
```

These are `y >= 0`, `y <= x + 1`, and `x + y <= 4`. They bound a non-empty triangle, so the answer is
`YES`.

Example 2:

```
2
1 0 1
-1 0 -2
```

These are `x <= 1` and `x >= 2`. No `x` satisfies both, so the answer is `NO`.

## Background

The question "is there a point satisfying all `m` linear inequalities?" looks like a job for a general
linear-programming solver, but two structural facts shape the approach:

- The intersection of half-planes is always a (possibly empty, possibly unbounded) **convex** region.
  A convex region with integer-coefficient bounding lines, if non-empty, always contains a point with
  bounded coordinates, so feasibility can be decided combinatorially rather than by a numeric optimizer.
- Two families of approach are on the table before committing:
  - **Pairwise vertex enumeration.** Every candidate "corner" of the feasible region is the
    intersection of two of the boundary lines; there are `O(m^2)` such points, and the region is
    non-empty iff one of them (or, for unbounded regions, a clipped variant) satisfies all `m`
    constraints. This is `O(m^3)` to test and is hopeless at `m = 2*10^5`, but it is trivially correct
    and makes a perfect oracle on small inputs.
  - **Incremental half-plane intersection.** Sort the half-planes by the angle of their boundary
    direction and sweep them into a double-ended queue, maintaining the running intersection boundary.
    Each half-plane is pushed once and popped at most once, giving `O(m log m)` after the sort. The
    open questions are the exact deque-pop rule and how emptiness / unboundedness are detected.

## Evaluation settings

Judged on hidden tests covering: non-empty bounded regions; empty intersections (including
opposite/anti-parallel half-planes whose feasible bands do not overlap); unbounded-but-non-empty
regions; degenerate feasible regions that are a single segment or a single point; duplicate and
same-direction half-planes (where only the tightest matters); large `m = 2*10^5` with coefficients near
`10^6` (so exact arithmetic must avoid overflow); and the empty input `m = 0`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    if (!(cin >> m)) return 0;
    for (int i = 0; i < m; i++) {
        long long a, b, c;
        cin >> a >> b >> c;
        // half-plane: a*x + b*y <= c
        // TODO: collect the half-planes.
    }

    // TODO: decide whether the intersection of all m half-planes is non-empty.
    bool feasible = true;

    cout << (feasible ? "YES" : "NO") << "\n";
    return 0;
}
```
