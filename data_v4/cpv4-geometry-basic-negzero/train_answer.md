**Problem.** From the origin `O = (0,0)` you are given `n` lattice beacons `P[0..n-1]` with possibly negative or zero coordinates. For an ordered pair `i < j`, the *signed sweep* is the cross product `sweep(i, j) = x_i*y_j - x_j*y_i` (twice the signed area of triangle `O, P[i], P[j]`). Print the maximum sweep over all `i < j`. If `n < 2` there is no pair — print `NONE`. The maximum sweep itself may be negative or zero.

**Key idea — exact integer enumeration.** With `n <= 2000` there are at most `~2*10^6` ordered pairs, so evaluate `sweep(i, j) = x_i*y_j - x_j*y_i` for every `i < j` and keep the running maximum. This is `O(n^2)`, uses only integer arithmetic (no `atan2`, no floating point), and respects the `i < j` index order by construction — no need to sort by angle, which would both reorder the legal pairs and introduce float ties on collinear / zero-coordinate beacons.

**Pitfalls.**
1. *Wrong base case (the core trap).* The running maximum must start at a sentinel below every achievable sweep (`LLONG_MIN`), **not** `0`. Initializing at `0` injects a phantom candidate worth `0`; on an all-clockwise instance where every sweep is negative — e.g. `(0,3),(3,0),(4,-1)` whose pairs are `-9, -12, -3` and whose true answer is `-3` — that phantom `0` wins and the code wrongly prints `0`. The identity for a `max` over a set is `-infinity`, and `0` is only correct when the problem floors the answer at `0`, which this one does not.
2. *No-pair corner.* `n = 0` and `n = 1` must print the literal `NONE`. Read the header first, then guard: a present `n = 0` still owes `NONE`, so the guard must *print*, not silently return.
3. *Overflow.* `|x|, |y| <= 10^6`, so a product reaches `10^12` and the difference `2*10^12`. Use `long long` for coordinates and the accumulator; `int` silently overflows on large coordinates. The `LLONG_MIN` sentinel is only compared/assigned, never added to, so it cannot underflow.

**Edge cases.** `n = 0` / `n = 1` -> `NONE`; all-clockwise pairs -> a negative maximum (handled by the `LLONG_MIN` base); collinear or zero-coordinate beacons -> a legitimate `0` sweep (distinct from `NONE`); duplicated points -> a real `0` sweep; near-maximum coordinates -> fits `long long`.

**Complexity.** `O(n^2)` time (`~2*10^6` pair evaluations at `n = 2000`), `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) { return 0; }              // no header at all -> nothing to do
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    if (n < 2) {                                 // no ordered pair i<j exists
        cout << "NONE" << "\n";
        return 0;
    }

    // best signed area*2 over ordered pairs i<j: cross(P[i],P[j]) = x_i*y_j - x_j*y_i.
    // Must start from a REAL pair, not 0, because every cross product can be negative.
    long long best = LLONG_MIN;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            long long cr = x[i] * y[j] - x[j] * y[i];
            if (cr > best) best = cr;
        }
    }

    cout << best << "\n";
    return 0;
}
```
