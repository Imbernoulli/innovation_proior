## Problem

Fix a square of side $s$, with the two top corners colored green and the two bottom corners colored red. Inside the square sit additional green and red points; the four corners plus the interior points form $g$ green points and $r$ red points in total, and no three of all these points are collinear. The points are given as integer coordinates $(x_i, y_i)$. Green points are numbered $1\dots g$ with the top-left corner $(0,s)$ being green #1 and the top-right $(s,s)$ green #2; red points are numbered $1\dots r$ with the bottom-left corner $(0,0)$ being red #1 and the bottom-right $(s,0)$ red #2. Interior points take the remaining indices in arbitrary order.

A segment may be drawn between two points only if (a) the two points have the **same color**, and (b) the new segment does **not cross** any previously drawn segment (sharing an endpoint is allowed; a proper interior crossing is not). Two points are in the same *component* if one can reach the other along drawn segments.

Draw exactly $g-1$ green segments so that all green points lie in a single component, and exactly $r-1$ red segments so that all red points lie in a single (separate) component, with the whole drawing remaining crossing-free. A valid wiring always exists; produce one.

**Input:** integer $g$; then $g$ lines of green coordinates; then integer $r$; then $r$ lines of red coordinates. **Output:** $(g-1)+(r-1)$ lines, each `i j c` naming the two endpoint indices and the color `c` (`g` or `r`); the order of lines and of endpoints within a line does not matter.

**Scale:** $3 \le g, r \le 50000$, $0 < s \le 2\times10^{8}$, integer coordinates up to $2\times10^{8}$ (orientation arithmetic must use 64-bit integers).

## Code framework

```python
import sys

def cross(ax, ay, bx, by, cx, cy):
    # twice the signed area of (A,B,C); exact in 64-bit ints
    return (ax - bx) * (cy - by) - (cx - bx) * (ay - by)

def sgn(v):
    return 1 if v > 0 else (-1 if v < 0 else 0)

def inside(A, B, C, X, P):
    # strict interior: same orientation sign on all three directed edges
    ax, ay = P[A]; bx, by = P[B]; cx, cy = P[C]; xx, xy = P[X]
    s1 = sgn(cross(ax, ay, bx, by, xx, xy))
    s2 = sgn(cross(bx, by, cx, cy, xx, xy))
    s3 = sgn(cross(cx, cy, ax, ay, xx, xy))
    return s1 == s2 and s2 == s3

def solve(P, g, r):
    out = []
    G1, G2 = ('g', 1), ('g', 2)      # top-left, top-right corners (green)
    R1, R2 = ('r', 1), ('r', 2)      # bottom-left, bottom-right corners (red)

    # TODO

    return out

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    g = int(next(it)); P = {}
    for i in range(1, g + 1):
        P[('g', i)] = (int(next(it)), int(next(it)))
    r = int(next(it))
    for i in range(1, r + 1):
        P[('r', i)] = (int(next(it)), int(next(it)))
    out = solve(P, g, r)
    sys.stdout.write('\n'.join(f"{a} {b} {c}" for a, b, c in out) + '\n')

if __name__ == "__main__":
    main()
```
