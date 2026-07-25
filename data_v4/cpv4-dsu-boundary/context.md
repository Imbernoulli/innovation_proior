# Counting star clusters after range-union bridge builds

## Research question

There are `n` stars in a line, numbered `1..n`. You are given `m` bridge-building operations. Each
operation is a pair `(l, r)` with `1 <= l <= r <= n` and means **build bridges so that every star in
the inclusive range `l, l+1, ..., r` becomes mutually connected**. Bridges are permanent and
operations may overlap, repeat, or be single points (`l == r`, which connects nothing new).

After all `m` operations have been applied, the stars partition into connected clusters (two stars are
in the same cluster if there is a chain of bridges between them; a star touched by no operation is its
own singleton cluster). **Output the number of connected clusters.**

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`). Then `m` lines follow, each with two integers `l` and `r`
  (`1 <= l <= r <= n`) describing one range-union operation.
- Output (stdout): a single line with the number of connected clusters after all operations.
- Time limit: 1 second. Memory: 256 MB.

Example: with `n = 8` and operations `(2,4), (4,6), (7,7)`, the answer is `4`. Operation `(2,4)` joins
`{2,3,4}`; `(4,6)` joins `{4,5,6}` and so merges with the previous group into `{2,3,4,5,6}`; `(7,7)`
is a single point and joins nothing. The clusters are `{1}`, `{2,3,4,5,6}`, `{7}`, `{8}` — four of them.

## Evaluation settings

Judged on hidden tests covering: single-point ranges (`l == r`) that must connect nothing; ranges that
touch at a shared endpoint versus ranges separated by a one-index gap; fully overlapping and repeated
ranges; the no-operation case (`m = 0`, answer `n`); `n = 1`; and large adversarial inputs with
`n = m = 2*10^5` and many long overlapping ranges.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // TODO: apply each range-union (l, r) efficiently, then count the
    //       connected clusters among stars 1..n.

    int comps = 0;

    cout << comps << "\n";
    return 0;
}
```
