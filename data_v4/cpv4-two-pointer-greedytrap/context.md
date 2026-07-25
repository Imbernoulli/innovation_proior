# Maximum number of two-person zipline carts

## Research question

A canyon-crossing zipline launches **carts that carry exactly two riders**. You are given the
weights of `n` people. A cart can be launched with two people if and only if the sum of their
weights is **at most** the cart's capacity `L`. Each person rides in at most one cart (a person may
also stay behind and ride in no cart). You want to **launch as many carts as possible**. Output that
maximum number of carts.

Equivalently: how many disjoint pairs `{i, j}` with `w[i] + w[j] <= L` can you form from the multiset
of weights? This is a maximum matching on the "compatibility graph", but the graph has a special
threshold structure (a pair is allowed purely based on a sum bound). It is the kind of pairing
subproblem that appears in load-balancing, ride-sharing, and "buddy up to cross" puzzles, so getting
the pairing rule and the integer width exactly right matters.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `L`
  (`0 <= n <= 2*10^5`, `1 <= L <= 4*10^9`); the second line has `n` integers `w[i]`
  (`1 <= w[i] <= 2*10^9`), whitespace-separated. When `n = 0` the weight line is empty/absent.
- Output (stdout): a single line with the maximum number of carts (pairs) that can be launched.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 4`, `L = 4`, `w = [1, 1, 2, 3]` the answer is `2` (pair `1+3 = 4` and `1+2 = 3`).

## Evaluation settings

Judged on hidden tests covering: arrays where every pair fits (answer `floor(n/2)`), arrays where no
pair fits (answer `0`), many ties near the limit, the empty array (`n = 0`), a single person
(`n = 1`, answer `0`), odd `n`, and large `n = 2*10^5` with weights near `2*10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // TODO: maximize the number of disjoint pairs {i, j} with w[i] + w[j] <= L.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
