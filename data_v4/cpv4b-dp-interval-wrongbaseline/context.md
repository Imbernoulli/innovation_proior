# Minimum splice cost for a circular film reel

## Research question

An archivist has `n` film reels arranged on a circular take-up carousel, in fixed clockwise
order. Reel `i` currently holds `w[i]` metres of film. To restore the movie the archivist must
splice all reels into a single continuous reel. A splice operation may only join **two reels that
are currently next to each other on the carousel** (the carousel is circular, so the last reel and
the first reel are also neighbours). Splicing two neighbouring reels of length `x` and `y` produces
one reel of length `x + y` that occupies their combined slot, and it costs `x + y` metres of leader
tape (the machine threads the whole combined length through once). After the splice the carousel has
one fewer reel, and the new reel's two outer ends become the new neighbours of whatever was beside
the originals.

The archivist repeats splices until a single reel remains. Different orders of adjacent splices give
different total tape costs. Compute the **minimum total cost** to merge all `n` reels into one.

The single-reel and empty carousels are corner cases worth nailing down.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 1000`), the number of reels; then `n` integers
  `w[i]` (`1 <= w[i] <= 10^6`), whitespace-separated, in clockwise carousel order.
- Output (stdout): a single line with the minimum total splicing cost.
- If `n <= 1` there is nothing to splice, so the cost is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `w = [4, 2, 6, 3, 5]` the answer is `46`.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1` (cost `0`); `n = 2`; small carousels checked
against brute force over all adjacent-merge orders (including the wrap pair); a range of carousel
configurations including skewed and uniform reel sizes; and large `n = 1000` with values near `10^6`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // TODO: compute the minimum total cost to splice all n reels on the circular
    // carousel into one, where each splice may only join two currently-adjacent reels.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
