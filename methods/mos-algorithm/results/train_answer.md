We are given a static array $a[0..n-1]$ and a batch of $q$ inclusive range queries $(l, r)$, and for each one we must report the number of distinct values in $a[l..r]$, with the answers printed back in the original query order. The naive solution is correct but wasteful: for each query we build $\texttt{set}(a[l..r])$ and take its size, which costs $O(n)$ per query and $O(nq)$ overall, and in the worst case — where every query spans most of the array — that is exactly $n\cdot q$ rescans of almost the same data. The whole inefficiency is that we throw away the work between consecutive queries even though neighboring ranges overlap heavily. Building a persistent or segment-tree structure for distinctness is possible but delicate; what we really want is to exploit the fact that all $q$ queries are known in advance (the problem is offline) and that adjacent ranges differ by only a few endpoint indices. That observation is what existing per-query rescans fail to use.

I propose Mo's algorithm. The core realization is that the distinct count is not something to recompute from scratch but something to maintain incrementally as we slide the range's endpoints. I keep an explicit current inclusive range $[\texttt{cur\_l}, \texttt{cur\_r}]$ whose contents are described exactly by a frequency table $\texttt{cnt}$ (how many times each value currently sits inside the range) and a single running integer $\texttt{distinct}$ (how many values have $\texttt{cnt} > 0$). The empty initial range is $\texttt{cur\_l}=0,\ \texttt{cur\_r}=-1$. The only events that change $\texttt{distinct}$ are zero crossings of a value's frequency, so adding and removing a single index are both $O(1)$:
$$\texttt{add}(i):\quad \texttt{cnt}[a_i] \mathrel{+}= 1,\quad \text{and } \texttt{distinct}\mathrel{+}=1 \text{ iff } \texttt{cnt}[a_i]=1,$$
$$\texttt{remove}(i):\quad \texttt{cnt}[a_i] \mathrel{-}= 1,\quad \text{and } \texttt{distinct}\mathrel{-}=1 \text{ iff } \texttt{cnt}[a_i]=0.$$
A value enters the distinct set precisely when its count rises from $0$ to $1$, and leaves precisely when it falls from $1$ to $0$; counts in between change nothing. Using a dictionary-backed $\texttt{cnt}$ rather than a fixed array keeps this correct for arbitrary integer values without any coordinate compression. To move from one target range to another I adjust the endpoints one index at a time — grow left by decrementing $\texttt{cur\_l}$ then $\texttt{add}(\texttt{cur\_l})$, grow right by incrementing $\texttt{cur\_r}$ then $\texttt{add}(\texttt{cur\_r})$, shrink left by $\texttt{remove}(\texttt{cur\_l})$ then incrementing $\texttt{cur\_l}$, shrink right by $\texttt{remove}(\texttt{cur\_r})$ then decrementing $\texttt{cur\_r}$ — and every step preserves the invariant, so once the endpoints reach $l$ and $r$ the value of $\texttt{distinct}$ is the answer, which I store at the query's original index.

What remains is the only nontrivial design choice: the order in which to visit the queries. If I process them as given, the endpoints can be dragged across the entire array after almost every query, and the movement is again $O(nq)$ — incremental updates alone buy nothing. Sorting solely by $r$ makes $\texttt{cur\_r}$ monotone but lets $\texttt{cur\_l}$ jump arbitrarily; sorting solely by $l$ has the mirror-image flaw. The fix is to keep one endpoint local while the other moves monotonically over a stretch. I pick a block width $\texttt{block}$, group the queries by which left-endpoint block they fall in, and within each group sort by increasing $r$ — concretely, I sort the query indices lexicographically by the key $(l \mathbin{//} \texttt{block},\ r)$. Inside one left-block the right endpoints are nondecreasing, so $\texttt{cur\_r}$ sweeps across the array at most once per block (plus one reset when the next block begins); with $O(n/\texttt{block})$ blocks this bounds total right-endpoint travel by $O(n^2/\texttt{block})$. The left endpoint, meanwhile, stays within an interval of width $\texttt{block}$ across consecutive queries of a group, so it moves $O(\texttt{block})$ per query, giving $O(q\cdot\texttt{block} + n)$ including the lower-order cross-block scans. The total pointer movement is therefore
$$O\!\left(\frac{n^2}{\texttt{block}} + q\cdot\texttt{block} + n\right),$$
and balancing the two main terms by setting $\texttt{block} = \lfloor\sqrt{n}\rfloor$ yields $O\big((n+q)\sqrt{n}\big)$. A larger block lets $\texttt{cur\_l}$ wander too far within a group; a smaller block creates too many groups and forces $\texttt{cur\_r}$ to restart too often — the square root is the equilibrium that turns repeated full rescans into bounded endpoint travel. Adding the $O(q\log q)$ sort gives an overall time of $O\big(q\log q + (n+q)\sqrt{n}\big)$ and space $O(q + d)$ where $d$ is the number of distinct values. A quick hand-check confirms the bookkeeping: for $a = [1,1,2,1,3]$, reaching $(0,2)$ from empty adds indices $0,1,2$ so $\texttt{cnt}[1]=2,\ \texttt{cnt}[2]=1,\ \texttt{distinct}=2$; moving to $(1,4)$ adds indices $3,4$ to bring in values $1,2,3$ and then removes index $0$, but since another $1$ remains, $\texttt{distinct}$ stays $3$ — matching the brute-force sets exactly, by the zero-crossing rule.

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
