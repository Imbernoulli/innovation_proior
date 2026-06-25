**Problem.** `n` climbers with weights `w[0..n-1]` must descend by a gondola whose cabin holds at most two climbers per trip with combined weight at most `C`; every `w[i] <= C`. Minimize the number of trips. Read `n` and `C`, then the `n` weights, from stdin; print the minimum trip count.

**Key idea — greedy-exchange, two-pointer.** Sort the weights ascending. Walk a left pointer `i` (lightest waiting) and a right pointer `j` (heaviest waiting). On each trip the heaviest waiting climber boards; if the lightest also fits (`w[i] + w[j] <= C`), send them together (`i++`), otherwise the heavy one rides alone. Either way `j--` and the trip count goes up. This runs while `i <= j`, so the lone middle climber (when `i == j`) is counted. The exchange argument: if the lightest climber `L` can pair with the heaviest `H` at all, pairing them is never worse, because any optimal plan can be rearranged so `L` rides with `H` without adding a trip; and if `L` cannot pair with `H`, then nobody can (everyone is at least as heavy as `L`), so `H` must ride alone — exactly what the greedy does. `O(n log n)`.

**Pitfalls.**
1. *The tempting closed-form is FALSE.* There are two valid lower bounds — the weight bound `ceil(sum(w)/C)` and the slot bound `ceil(n/2)` — and it is tempting to assert the answer is their maximum. Check it numerically before trusting it: on `w = [1,1,2,5,5,5]`, `C = 5`, the weight bound is `ceil(19/5) = 4`, the slot bound is `ceil(6/2) = 3`, so the combined bound is `4`; but the true answer (greedy, or exhaustive) is `5`, because each of the three `5`-kg climbers wastes a sharable slot that neither bound can see. Lower bounds are not the answer — simulate the greedy.
2. *Off-by-one at the meeting point.* Loop while `i <= j`, not `i < j`; otherwise the single remaining climber (when `i == j`) is never counted, e.g. `w = [3]` would print `0` instead of `1`. Inside the body, guard the pairing with `i != j` so you never "pair" the last lone climber with himself.
3. *Overflow.* A pair sum `w[i] + w[j]` reaches `2*10^9`, past 32-bit `int`; keep `C`, the weights, and the comparison in `long long`. An `int` pair-sum wraps negative and makes the capacity test pass when it should fail.

**Edge cases.** `n = 0` -> `0` (empty loop; absent input prints nothing via the `cin` guard). `n = 1` -> `1`. Everyone too heavy to pair -> `n`. Everyone pairs -> `ceil(n/2)`. Two weights summing exactly to `C` share a trip (use `<=`, since "must not exceed" allows equality).

**Complexity.** `O(n log n)` time (the sort), `O(1)` extra space beyond the input.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;          // empty input -> no climbers -> 0 trips
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    sort(w.begin(), w.end());                // greedy-exchange needs sorted weights

    // Two-pointer: each trip carries the heaviest remaining climber; if the
    // lightest remaining one also fits under capacity C, send the two together.
    long long trips = 0;
    int i = 0, j = n - 1;
    while (i <= j) {
        if (i != j && w[i] + w[j] <= C) {    // pair lightest with heaviest when they fit
            i++;
        }
        j--;                                  // heaviest always boards this trip
        trips++;
    }

    cout << trips << "\n";
    return 0;
}
```
