# Minimum-cost XOR merging of a row of stones

## Research question

You are given `n` stones in a row; stone `i` carries a non-negative integer value `a[i]`. A *merge*
takes two **currently adjacent** stones with values `x` (left) and `y` (right), removes them, and puts
in their place a single new stone whose value is `x XOR y` (bitwise exclusive-or). The merge costs
`x OR y` (bitwise or) — the OR is computed on the two values *at the moment they are merged*, not on
the originals. You perform merges until one stone remains; that takes exactly `n - 1` merges.

Among all `n - 1` choices of which adjacent pair to merge at each step, output the **minimum possible
total cost**. Different merge orders produce different intermediate values, so the cost depends on the
order even though — as it happens — the final stone's value does not.

This is an interval / chain problem in the family of "merge adjacent things, pay a cost that depends
on what you merge" (matrix-chain, optimal BST, stone-merging). The wrinkle here is that the cost is a
*bitwise* function, which makes it tempting to look for a slick closed form and dangerous to trust one.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 1500`); then `n` integers `a[i]`
  (`0 <= a[i] < 2^20`), whitespace-separated.
- Output (stdout): a single line with the minimum total merge cost. If `n <= 1` no merge happens, so
  the cost is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `a = [6, 5, 3]` the answer is `10`.
Merging the left pair first costs `(6 OR 5) + ((6 XOR 5) OR 3) = 7 + (3 OR 3) = 7 + 3 = 10`. Merging
the right pair first costs `(5 OR 3) + (6 OR (5 XOR 3)) = 7 + (6 OR 6) = 7 + 6 = 13`. The minimum is
`10`.

## Background

Two observations sit at the surface and invite a shortcut; both must be treated with suspicion until
checked, because a wrong bitwise identity asserted here propagates silently into every test.

- **The final value is order-independent.** Because XOR is associative and commutative, the last
  remaining stone always equals `a[0] XOR a[1] XOR ... XOR a[n-1]`, no matter how you merge. It is
  very tempting to conclude from this that the *cost* is also order-independent or has a one-line
  closed form (e.g. some multiple of the OR of all values, or a fixed sum of adjacent ORs). That
  conclusion does not follow and is the trap this problem is built around.
- **A merge only ever combines two contiguous blocks.** At any moment the surviving stones correspond
  to a partition of the original row into contiguous segments, and a stone's value is the XOR of the
  original values in its segment. So a full merge sequence is exactly a way of parenthesizing the row,
  and the cost of the final merge that joins a left block `[l..k]` with a right block `[k+1..r]` is
  `(XOR of a[l..k]) OR (XOR of a[k+1..r])`. This is the structure an interval DP exploits.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1` (cost `0`), `n = 2`, all-zero rows (cost `0`),
rows engineered so a naive left-to-right or greedy "merge the cheapest adjacent pair" order is
strictly worse than optimal, rows where the order-independent final value tempts a false closed form,
and large `n = 1500` with values near `2^20` so the total cost exceeds 32 bits and the chosen
algorithm must fit the time limit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the minimum total cost of merging the row to one stone,
    //       where merging adjacent values x (left) and y (right) costs (x OR y)
    //       and produces the new value (x XOR y).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
