## Problem

Fix a square of side $s$, with the two top corners colored green and the two bottom corners colored red. Inside the square sit additional green and red points; the four corners plus the interior points form $g$ green points and $r$ red points in total, and no three of all these points are collinear. The points are given as integer coordinates $(x_i, y_i)$. Green points are numbered $1\dots g$ with the top-left corner $(0,s)$ being green #1 and the top-right $(s,s)$ green #2; red points are numbered $1\dots r$ with the bottom-left corner $(0,0)$ being red #1 and the bottom-right $(s,0)$ red #2. Interior points take the remaining indices in arbitrary order.

A segment may be drawn between two points only if (a) the two points have the **same color**, and (b) the new segment does **not cross** any previously drawn segment (sharing an endpoint is allowed; a proper interior crossing is not). Two points are in the same *component* if one can reach the other along drawn segments.

Draw exactly $g-1$ green segments so that all green points lie in a single component, and exactly $r-1$ red segments so that all red points lie in a single (separate) component, with the whole drawing remaining crossing-free. A valid wiring always exists; produce one.

**Input:** integer $g$; then $g$ lines of green coordinates; then integer $r$; then $r$ lines of red coordinates. **Output:** $(g-1)+(r-1)$ lines, each `i j c` naming the two endpoint indices and the color `c` (`g` or `r`); the order of lines and of endpoints within a line does not matter.

**Scale:** $3 \le g, r \le 50000$, $0 < s \le 2\times10^{8}$, integer coordinates up to $2\times10^{8}$ (orientation arithmetic must use 64-bit integers).

## Code framework

The deliverable is a single self-contained C++17 program. It reads `g`, the `g` green coordinates, then `r`, the `r` red coordinates from stdin, and writes exactly `(g-1)+(r-1)` segment lines `i j c` to stdout, where `c` is `g` or `r`.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int g;
    if (!(cin >> g)) return 0;

    vector<pair<long long, long long>> green(g + 1);
    for (int i = 1; i <= g; ++i) {
        cin >> green[i].first >> green[i].second;
    }

    int r;
    cin >> r;

    vector<pair<long long, long long>> red(r + 1);
    for (int i = 1; i <= r; ++i) {
        cin >> red[i].first >> red[i].second;
    }

    vector<tuple<int, int, char>> output;

    // TODO: fill output with segment endpoints and their color.

    for (auto [i, j, c] : output) {
        cout << i << ' ' << j << ' ' << c << '\n';
    }

    return 0;
}
```
