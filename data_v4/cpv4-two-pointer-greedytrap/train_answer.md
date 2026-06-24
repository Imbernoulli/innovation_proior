**Problem.** Given `n` people with weights `w[0..n-1]` and a cart capacity `L`, a two-seat cart launches iff the two riders' weights sum to at most `L`. Each person rides at most once. Maximize the number of carts (disjoint pairs `{i, j}` with `w[i] + w[j] <= L`). Read `n`, `L`, then the weights from stdin; print the maximum cart count.

**Why the obvious greedy is wrong.** "Sort, then repeatedly pair the two lightest people" fails because the objective is the *number* of pairs, and a light person's real value is as a partner for a heavy person who otherwise cannot ride at all. On `[1, 1, 2, 3]` with `L = 4`, the lightness-greedy marries the two `1`s (`1+1=2`) for **1** cart and strands `2` and `3` (`2+3=5>4`); but `1+3=4` and `1+2=3` give **2** carts. Spending light people on each other wastes them. Greedy is discarded.

**Key idea — two pointers from both ends.** Sort ascending. Put `lo` at the lightest free person, `hi` at the heaviest. Repeatedly:

- if `w[lo] + w[hi] <= L`: pair them (`pairs++`, `lo++`, `hi--`);
- else: the heaviest cannot ride with anyone (everyone else is `>= w[lo]`, so every sum involving `hi` exceeds `L`), so discard it (`hi--`).

Stop when `lo >= hi`. The count is optimal.

**Correctness.** The discard is forced: if even the lightest cannot lift the heaviest, the heaviest is unpaired in *every* feasible solution, so dropping it loses nothing. The pairing is safe by exchange: in any optimal solution where `lo` and `hi` are paired with `q` and `p` respectively, re-pair into `{lo, hi}` and `{p, q}`; feasibility of `{p, q}` holds because `w[q] <= w[hi]` (hi is heaviest) gives `w[p] + w[q] <= w[p] + w[hi] <= L`. So an optimal solution pairing `{lo, hi}` always exists; induct. The crux is pairing from the *heavy* end — the heaviest person is the most constrained, so it deserves the lightest available partner.

**Pitfalls.**
1. *Wrong greedy direction.* Pairing the two lightest (or the two closest) is a count-losing local rule; pair lightest-with-heaviest. The instance `[1,1,2,3]`, `L=4` separates them (1 vs 2 carts).
2. *Self-pairing off-by-one.* Loop on `lo < hi`, not `lo <= hi`. With `lo == hi` both pointers name the same person and `w[lo] + w[hi]` double-counts one weight; an odd-sized all-fit group then reports a phantom extra cart (a single person would return 1 instead of 0).
3. *Overflow.* `L` reaches `4*10^9` and a pair-sum reaches `2*10^9 + 2*10^9 = 4*10^9`, both past 32-bit `INT_MAX`. Store `L` and the weights as `long long`; an `int` element type overflows the sum *before* the comparison widens, a silent wrong-answer.

**Edge cases.** `n = 0` → `0` (loop never runs, `hi = -1` only used in the guard). `n = 1` → `0`. All pairs fit → `floor(n/2)` (the absolute max). No pair fits → `0`. Odd `n` leaves a correct leftover.

**Complexity.** `O(n log n)` for the sort plus `O(n)` for the sweep; `O(1)` extra space beyond the input.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    sort(w.begin(), w.end());

    // Two pointers from both ends: pair the lightest free rider with the
    // heaviest free rider whenever they fit together (sum <= L). If they do
    // not fit, the heaviest cannot ride with anyone (everyone else is >= the
    // lightest), so drop it and keep trying with the next-heaviest.
    int lo = 0, hi = n - 1;
    long long pairs = 0;
    while (lo < hi) {
        if (w[lo] + w[hi] <= L) {
            pairs++;
            lo++;
            hi--;
        } else {
            hi--;
        }
    }

    cout << pairs << "\n";
    return 0;
}
```
