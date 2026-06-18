# Mo's algorithm

## Problem

Given a static array `a[0..n-1]` and inclusive queries `(l, r)`, return the
number of distinct values in `a[l..r]` for each query, in the original query
order.

## Method

Maintain an inclusive range `[cur_l, cur_r]`, a frequency table `cnt`, and a
running count `distinct`. When index `i` enters the range, increment
`cnt[a[i]]`; if the count becomes `1`, increment `distinct`. When index `i`
leaves the range, decrement `cnt[a[i]]`; if the count becomes `0`, decrement
`distinct`. Each endpoint step costs `O(1)` expected time.

Choose `block = max(1, floor(sqrt(n)))`. Sort query indices lexicographically by

```text
(l // block, r)
```

where `(l, r)` is the query for that index. Within one left-endpoint block, the
right endpoint moves monotonically except for the reset between blocks, giving
`O(n^2 / block)` right-endpoint movement. The left endpoint moves `O(block)` per
query, plus lower-order cross-block movement, giving `O(q * block + n)`. With
`block = sqrt(n)`, total pointer movement is `O((n + q) sqrt(n))`; sorting costs
`O(q log q)`.

## Code

```python
import sys
from collections import defaultdict


def answer_queries(a, queries):
    """Return one distinct-count answer per inclusive 0-based query."""
    n = len(a)
    cnt = defaultdict(int)
    distinct = 0
    answers = [0] * len(queries)

    def add(i):
        nonlocal distinct
        value = a[i]
        cnt[value] += 1
        if cnt[value] == 1:
            distinct += 1

    def remove(i):
        nonlocal distinct
        value = a[i]
        cnt[value] -= 1
        if cnt[value] == 0:
            distinct -= 1

    block = max(1, int(n ** 0.5))
    order = sorted(
        range(len(queries)),
        key=lambda idx: (queries[idx][0] // block, queries[idx][1]),
    )
    cur_l, cur_r = 0, -1

    def place_answer(idx):
        nonlocal cur_l, cur_r
        l, r = queries[idx]
        while cur_l > l:
            cur_l -= 1
            add(cur_l)
        while cur_r < r:
            cur_r += 1
            add(cur_r)
        while cur_l < l:
            remove(cur_l)
            cur_l += 1
        while cur_r > r:
            remove(cur_r)
            cur_r -= 1
        answers[idx] = distinct

    for idx in order:
        place_answer(idx)
    return answers


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    q = int(next(it))
    queries = [(int(next(it)) - 1, int(next(it)) - 1) for _ in range(q)]
    out = answer_queries(a, queries)
    sys.stdout.write("\n".join(map(str, out)))


if __name__ == "__main__":
    main()
```

## Complexity

Time is `O(q log q + (n + q) sqrt(n))`. Space is `O(q + d)`, where `d` is the
number of distinct values in `a`.
