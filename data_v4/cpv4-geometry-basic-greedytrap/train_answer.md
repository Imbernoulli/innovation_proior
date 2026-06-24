**Problem.** A road is the real line with `n` houses at integer positions `x[0..n-1]`. A lamp placed at any real `s` lights the closed segment `[s, s+L]`. Cover every house with the fewest length-`L` lamps and print the count. Boundaries count (closed segment), positions may repeat, and `n = 0` answers `0`. Read `n`, `L`, then the `x[i]` from stdin.

**Why the obvious greedy is wrong.** "Sort the houses, then center a lamp on the leftmost still-dark house" feels symmetric but is suboptimal: centering spends half the lamp's length, `L/2`, on empty road to the *left* of the leftmost dark house, where nothing needs lighting. That wasted reach could have stretched rightward. On `[2, 3, 9, 9, 14, 20]` with `L = 5`, centering uses **4** lamps (the lamp at the `9` cluster, window `[6.5, 11.5]`, cannot also reach `14`), but anchoring each lamp's **left edge** at the leftmost dark house uses **3**: `[2,7]`, `[9,14]` (catches `14`), `[20,25]`. The centered greedy is discarded.

**Key idea — leftmost-dark, left-anchored greedy.** Sort the houses. While houses remain, let `p` be the leftmost still-dark house and place one lamp covering `[p, p+L]`; skip every house with `x[i] <= p + L`; repeat. Optimality is an exchange argument: any lamp covering `p` has left edge `s <= p`, so right edge `s + L <= p + L`; the choice `s = p` maximizes the right edge among all `p`-covering lamps, so `[p, p+L]` covers a superset of the right-side houses any other valid lamp could. Replacing an optimal solution's first lamp by `[p, p+L]` never increases the count; induct on the rest.

**Pitfalls to get right.**
1. *Anchoring rule.* Put the lamp's **left edge** at the leftmost dark house, not its center and not its right edge. Centering is the trap above; right-aligning would fail to cover `p` at all.
2. *Out-of-bounds skip.* The inner skip loop dereferences `x[i]` after advancing, so it must short-circuit `i < n && x[i] <= right`. Without the `i < n` test, consuming the last house reads past the array (a trace of single house `[5]`, `L=3` exposes it).
3. *Closed boundary.* Use `x[i] <= right` (not `<`); a house exactly at `p + L` is lit and must be absorbed, else `[0,5]` with `L=5` wrongly splits into two lamps.
4. *Overflow.* `x[i] + L` reaches `10^9 + 2*10^9 = 3*10^9`, past 32-bit range; keep `x`, `L`, and `right` in `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `n = 0` -> `0`; `n = 1` -> `1`; `L = 0` -> number of distinct positions (each lamp lights a single point, duplicates at one spot share a lamp); huge `L` -> `1`; negative coordinates behave identically (the trap reappears for negatives, the left-anchor rule still wins).

**Complexity.** `O(n log n)` for the sort, then a single linear sweep; `O(1)` extra space. Comfortable for `n = 2*10^5` within 1 second.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;          // empty input -> nothing to do
    vector<long long> x(n);
    for (auto &v : x) cin >> v;

    if (n == 0) {                            // no houses -> no lamps needed
        cout << 0 << "\n";
        return 0;
    }

    sort(x.begin(), x.end());

    long long lamps = 0;
    int i = 0;
    while (i < n) {
        // The leftmost still-dark house is x[i]. Anchor a lamp's LEFT edge at x[i]
        // so it covers [x[i], x[i] + L]; this reaches as far right as any lamp can
        // while still covering x[i]. Cover every house inside that interval.
        long long right = x[i] + L;
        lamps++;
        while (i < n && x[i] <= right) i++;
    }

    cout << lamps << "\n";
    return 0;
}
```
