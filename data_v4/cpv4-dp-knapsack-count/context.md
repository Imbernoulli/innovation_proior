# Counting bakery boxes: subsets with an exact budget and a joy floor

## Research question

A bakery lists `n` pastries on its menu. Pastry `i` has an integer `price[i]` and an integer
`joy[i]`. You want to assemble a *gift box* by choosing a **subset** of the menu — each pastry can be
put in the box **at most once** (there is one of each on display). A box is *valid* when:

- its total price is **exactly** equal to your budget `B` (you must spend the whole budget, no more, no
  less), and
- its total joy is **at least** a threshold `J`.

Count how many **distinct valid boxes** exist and output that count **modulo `1000000007`**. Two boxes
are the same if and only if they use the same set of pastries; the empty box is a legitimate box (it
has price `0` and joy `0`), so it is counted whenever `B = 0` and `J = 0`.

This is a two-dimensional 0/1 knapsack **counting** problem. The thing that makes it more than a
textbook exercise is that the count is extremely sensitive to *how* the knapsack is rolled up: a single
wrong loop direction silently turns "subsets" into "sequences" and over-counts, and a single wrong
comparison turns "joy at least `J`" into "joy at least `J-1`" and over-counts again. Getting the index,
the modulus reduction, and the threshold bucket all exactly right is the whole task.

## Input / output contract

- Input (stdin): the first line has three integers `n B J`
  (`0 <= n <= 100`, `0 <= B <= 1500`, `0 <= J <= 1500`).
  Then follow `n` lines, the `i`-th containing two integers `price[i] joy[i]`
  (`0 <= price[i] <= 1500`, `0 <= joy[i] <= 1500`).
- Output (stdout): a single line with the number of distinct valid boxes, taken modulo `1000000007`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the menu

```
4 6 7
2 5
4 4
2 3
4 6
```

the answer is `4`. Spending exactly `6` and collecting joy `>= 7` can be done with the pastry sets
`{0,1}` (price `6`, joy `9`), `{1,2}` (price `6`, joy `7`), `{0,3}` (price `6`, joy `11`), and `{2,3}`
(price `6`, joy `9`) — four distinct boxes.

## Background

The constraint "total price *exactly* `B`" with "each item at most once" is a 0/1 subset-sum, and we
are *counting* solutions, not just deciding feasibility. Two design questions are open before
committing to an implementation:

- **What the second dimension is.** The joy condition is a *lower* bound, not an exact value, so naively
  carrying the exact total joy would need a dimension as large as the sum of all joys. The open question
  is whether we can compress every "joy `>= J`" outcome into one bucket so the second dimension stays
  `O(J)`, and what the exact comparison/clamp for that bucket must be.
- **How to roll up the items so subsets are counted once.** A 1-D 0/1 knapsack is classically rolled up
  in place over the capacity axis, but the *direction* of that sweep is exactly what separates "each
  item used at most once" (a subset) from "each item reused freely" (a multiset / ordered fill). With a
  *count* in the cell rather than a boolean, choosing the wrong direction does not crash — it returns a
  larger, wrong number.

## Evaluation settings

Judged on hidden tests covering: `B = 0` and `J = 0` (the empty box counts); items whose price exceeds
`B` (must be ignored cleanly); price-`0` and joy-`0` items (which create many same-price boxes and
stress the de-duplication); the joy threshold landing exactly on a box's joy (the `>=` boundary);
menus where many subsets collide on the same price (so a double-count would be visible); and full-size
`n = 100`, `B = J = 1500` instances where the count is large and must be reduced modulo `1000000007`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long B, J;
    if (!(cin >> n >> B >> J)) return 0;

    const long long MOD = 1000000007LL;

    vector<long long> price(n), joy(n);
    for (int i = 0; i < n; i++) cin >> price[i] >> joy[i];

    // TODO: count distinct subsets with total price == B and total joy >= J, mod MOD.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
