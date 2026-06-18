# Context

## Problem

Given a static array of `n` integers and `q` offline queries `(l, r)`, report the
number of distinct values in `a[l..r]`. (`n, q` up to ~10^5 / 10^6.)

All `q` queries are available before any answer has to be printed, and the output
must still follow the original query order. Inside `answer_queries`, indices are
0-based and each range includes both endpoints; the input wrapper reads 1-based
query endpoints and converts them.

## Code framework

The array and the query list are read in. A top-level `answer_queries(a, queries)`
returns one integer per query, in the order the queries were given. A frequency
table `cnt` indexed by value, together with a running `distinct` count, is
available; the `add(i)` / `remove(i)` hooks are the single place where bringing one
more index into, or out of, the collection currently being tracked updates that
running count.

```python
import sys
from collections import defaultdict


def answer_queries(a, queries):
    """Return one distinct-count answer per inclusive 0-based query."""
    n = len(a)
    cnt = defaultdict(int)          # how many times each value is in play
    distinct = 0                    # values currently with count > 0
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

    order = range(len(queries))

    def place_answer(idx):
        # TODO
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
