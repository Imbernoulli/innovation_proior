**Problem.** Given `n` points with integer coordinates (`-10^9 <= x, y <= 10^9`, `n <= 500`, points may coincide), find the maximum-area triangle over all triples of distinct points and print **twice** that area as an exact integer. If `n < 3` or every triple is degenerate, print `0`. Read from stdin, write to stdout.

**Key idea — brute force over all triples with exact integer arithmetic.** Twice the signed area of triangle `ABC` is the 2-D cross product

```
cross = (Bx - Ax) * (Cy - Ay) - (Cx - Ax) * (By - Ay),
```

and twice the *area* is `|cross|`, an integer for integer inputs (no division, no floating point). The maximum-area triangle is the triple maximizing `|cross|`. With `n <= 500` there are at most `~2.1 * 10^7` triples, so an `O(n^3)` scan keeping the maximum `|cross|` runs well inside 2 seconds and is obviously correct. Reporting twice the area (not the area) is deliberate: the area itself is a half-integer near `4 * 10^18`, which would tempt a `double` that cannot represent it exactly (53-bit mantissa, exact only to `~9 * 10^15`).

**Pitfalls.**
1. *32-bit overflow — the whole problem.* Coordinate differences reach `2 * 10^9` and a single product `(Bx-Ax)*(Cy-Ay)` reaches `4 * 10^18`; the cross product reaches `~8 * 10^18`. That fits `long long` (max `~9.22 * 10^18`) but is billions of times too large for `int`. Computing the cross product in `int`, e.g. on `A=(0,0), B=(10^9,0), C=(0,10^9)`, gives `-1486618624` instead of `10^18` — wrong magnitude *and* sign, and it fails *silently* (no crash, passes small tests, breaks the large hidden ones). Coordinates, differences, the product, `cross`, and the accumulator must all be `long long`.
2. *Truncating `abs`.* `abs`/`std::abs(int)` returns `int`; calling it on a `long long cross` can truncate a correct 64-bit value back to 32 bits, re-introducing overflow downstream of correct arithmetic. Use `llabs` (unambiguously `long long -> long long`).

**Edge cases (all covered by the loop + `best = 0` init):** `n < 3` -> no triple runs, output `0`; all-collinear or coincident points -> every `cross = 0`, output `0`; the four corners of a `\pm 10^9` square -> twice-area `4 * 10^18`, the value that an `int` could never hold.

**Complexity.** `O(n^3)` time, `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    // best holds twice the maximum triangle area (a non-negative integer).
    // With |coord| <= 1e9, edge differences reach 2e9 and the cross-product
    // products reach ~4e18, so all of this MUST be long long.
    long long best = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            for (int k = j + 1; k < n; k++) {
                long long abx = x[j] - x[i];
                long long aby = y[j] - y[i];
                long long acx = x[k] - x[i];
                long long acy = y[k] - y[i];
                long long cross = abx * acy - acx * aby; // = 2 * signed area
                long long twiceArea = llabs(cross);
                if (twiceArea > best) best = twiceArea;
            }

    cout << best << "\n";
    return 0;
}
```
