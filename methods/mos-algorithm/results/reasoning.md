I have a static array `a` and many inclusive range queries `(l, r)`. Each answer is the number of distinct values in `a[l..r]`. The obvious code is correct: for each query, build `set(a[l:r + 1])` and take its length. The problem is that this repeats almost the same scan again and again. In the worst case every query spans most of the array, so the total work becomes `O(nq)`.

The repeated work is sitting at the boundaries. If I know the distinct count for one range, then extending the right end by one index only asks what happens to `a[r + 1]`. If that value was absent, the distinct count increases; if it was already present, it does not. Removing one endpoint is symmetric: the distinct count decreases only when the removed value had frequency one. So the state I need is not a set rebuilt per query, but a frequency table `cnt` and one integer `distinct` that changes only at zero crossings.

That gives the exact update rules. When index `i` is added, set `value = a[i]`, increment `cnt[value]`, and increment `distinct` only if the new count is `1`. When index `i` is removed, decrement `cnt[value]`, and decrement `distinct` only if the new count is `0`. A dictionary-backed `cnt` keeps the code valid for arbitrary integer values in `a`.

Now I can move from one requested range to another by changing endpoints one step at a time. I keep an inclusive range `[cur_l, cur_r]` whose contents are exactly described by `cnt` and `distinct`; the empty initial range is `cur_l = 0, cur_r = -1`. To reach a target `(l, r)`, I grow left by decrementing `cur_l` before `add(cur_l)`, grow right by incrementing `cur_r` before `add(cur_r)`, shrink left by `remove(cur_l)` before incrementing `cur_l`, and shrink right by `remove(cur_r)` before decrementing `cur_r`. Each step preserves the invariant, so when the endpoints equal `l` and `r`, `distinct` is the answer for that query.

The remaining question is the order of the queries. If I process them as given, `cur_l` and `cur_r` can be dragged across the whole array after almost every answer, so the movement can still be `O(nq)`. Since all queries are already known, I can choose a better visiting order and store each result in `answers[idx]`.

Sorting only by `r` makes `cur_r` behave well but allows `cur_l` to jump anywhere. Sorting only by `l` has the same flaw with the other endpoint. I need to keep one endpoint local while the other moves monotonically for a while. Let `block` be a width. I group queries by `l // block`, and inside each group I sort by increasing `r`; in code, query indices are sorted lexicographically by `(queries[idx][0] // block, queries[idx][1])`.

For one left-endpoint group, the right endpoints are nondecreasing, so `cur_r` moves across the array at most once, plus a possible reset when the next group starts. There are `O(n / block)` groups, so right-endpoint movement is `O(n^2 / block)`. For the left endpoint, consecutive targets inside one group both have `l` values in an interval of width `block`, so `cur_l` moves `O(block)` per query. Cross-group movement contributes only a linear scan across group boundaries, and the useful bound is `O(q * block + n)` for the left endpoint. The total movement is therefore `O(n^2 / block + q * block + n)`.

Balancing the two main terms with `block = sqrt(n)` gives `O(n sqrt(n) + q sqrt(n) + n)`, which is `O((n + q) sqrt(n))` for `n >= 1`. Larger blocks make `cur_l` wander too much inside a group; smaller blocks create too many groups and make `cur_r` restart too often. The square-root block size is the balance that turns the repeated rescans into bounded endpoint travel. The sorting itself costs `O(q log q)`.

Let me check the count maintenance by hand. For `a = [1, 1, 2, 1, 3]`, reaching `(0, 2)` from the empty range adds indices `0`, `1`, and `2`, so `cnt[1] = 2`, `cnt[2] = 1`, and `distinct = 2`. Moving to `(1, 4)` adds indices `3` and `4`, making values `1`, `2`, and `3` present, then removes index `0`; because another `1` remains, `distinct` stays `3`. That matches the brute-force sets for both ranges, and the reason is exactly the zero-crossing rule for `cnt`.

Now the code can be written directly from those invariants. The `place_answer(idx)` helper moves `[cur_l, cur_r]` to `queries[idx]` and writes the answer back at the original index.

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

The full chain is: exact frequency maintenance makes each endpoint step constant time; an invariant range `[cur_l, cur_r]` lets answers be transferred between queries; lexicographic order by `(l // block, r)` controls the total endpoint travel; `block = sqrt(n)` gives `O((n + q) sqrt(n))` pointer movement after the `O(q log q)` sort.
