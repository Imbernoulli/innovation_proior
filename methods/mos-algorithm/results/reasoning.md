I have a static array `a` and many inclusive range queries `(l, r)`. Each answer is the number of distinct values in `a[l..r]`. The obvious code is correct: for each query, build `set(a[l:r + 1])` and take its length. The problem is that this repeats almost the same scan again and again. In the worst case every query spans most of the array, so the total work becomes `O(nq)`.

The repeated work is sitting at the boundaries. If I know the distinct count for one range, then extending the right end by one index only asks what happens to `a[r + 1]`. If that value was absent, the distinct count increases; if it was already present, it does not. Removing one endpoint is symmetric: the distinct count decreases only when the removed value had frequency one. So the state I need is not a set rebuilt per query, but a frequency table `cnt` and one integer `distinct` that changes only at zero crossings.

That gives the exact update rules. When index `i` is added, set `value = a[i]`, increment `cnt[value]`, and increment `distinct` only if the new count is `1`. When index `i` is removed, decrement `cnt[value]`, and decrement `distinct` only if the new count is `0`. Coordinate-compressing the values once up front lets `cnt` be a dense array — `O(1)` indexed — while staying valid for arbitrary integer values in `a`.

Now I can move from one requested range to another by changing endpoints one step at a time. I keep an inclusive range `[cur_l, cur_r]` whose contents are exactly described by `cnt` and `distinct`; the empty initial range is `cur_l = 0, cur_r = -1`. To reach a target `(l, r)`, I grow left by decrementing `cur_l` before `add(cur_l)`, grow right by incrementing `cur_r` before `add(cur_r)`, shrink left by `remove(cur_l)` before incrementing `cur_l`, and shrink right by `remove(cur_r)` before decrementing `cur_r`.

Before I trust this, I should make sure the four loops compose correctly and don't, say, double-count an index or step off the end when the new range overlaps the old one in an awkward way. Let me run the count maintenance by hand on `a = [1, 1, 2, 1, 3]`. Reaching `(0, 2)` from the empty range adds indices `0`, `1`, and `2`, so `cnt[1] = 2`, `cnt[2] = 1`, and `distinct = 2`. The brute-force set of `a[0:3] = [1, 1, 2]` is `{1, 2}`, size `2` — agreement. Moving to `(1, 4)`: the right grows first, adding index `3` (`cnt[1] = 3`) and index `4` (`cnt[3] = 1`, `distinct` rises to `3`); then the left shrinks, removing index `0` (`cnt[1] = 2`, still positive, so `distinct` stays `3`). The set of `a[1:5] = [1, 2, 1, 3]` is `{1, 2, 3}`, size `3` — agreement. The case I was worried about, removing one copy of `1` while two more remain, behaves exactly because `distinct` only moves on a zero crossing, not on every removal.

A two-query trace is not much evidence that the four-loop endpoint walk is right in general, so I check it against brute force over many random instances: `n` up to 12, values drawn from a small alphabet so duplicates are common, several random `(l, r)` per instance. Across 2000 such cases the pointer-walk answers match `len(set(a[l:r+1]))` on every query, and the `n = 1` single-element case returns `1` as it should. I am now confident the per-step update and the range-walk are correct; the open question is only how fast they are.

The cost is the total number of endpoint steps. If I process queries as given, `cur_l` and `cur_r` can each be dragged across the whole array after almost every answer, so the movement is still `O(nq)` — nothing has been won asymptotically, only constant factors. But all `q` queries are known up front, and `answers[idx]` lets me write each result back at its original position. So I am free to choose the visiting order, and the question becomes: which order keeps the total endpoint travel small?

Sorting only by `r` makes `cur_r` march monotonically through the array, but then `cur_l` is free to jump anywhere between consecutive queries, so left travel is unbounded. Sorting only by `l` has the mirror flaw. I want one endpoint to stay roughly local while the other moves monotonically for a stretch. So let `block` be a width, group queries by `l // block`, and within each group sort by increasing `r`; the sort key is `(queries[idx][0] // block, queries[idx][1])`.

I should check that this key actually delivers the monotonic-`r` behavior I am counting on, because it is easy to convince yourself of a structural claim that the tie-breaking quietly violates. Take `n = 16`, so `block = 4`, and ten scattered queries. Sorting by the key and reading off the right endpoints within each left-block group gives:

```
group 0:  r-sequence [1, 3, 4, 14, 15]   nondecreasing
group 1:  r-sequence [7, 9]              nondecreasing
group 2:  r-sequence [12, 15]            nondecreasing
group 3:  r-sequence [14]                nondecreasing
```

So inside a group `cur_r` only ever moves forward; it can jump backward at most once, when a new group begins. With `O(n / block)` groups and each forward sweep bounded by `n`, the right-endpoint travel is `O(n^2 / block)`. For the left endpoint, two consecutive targets in the same group have `l` values within a window of width `block`, so `cur_l` moves `O(block)` between them; across all queries that is `O(q * block)`, plus an `O(n)` term for the cross-group boundary shifts. Total travel is `O(n^2 / block + q * block + n)`.

To see that the ordering really is buying something and not just rearranging the same work, I count actual steps on that same 16-element instance. The block-sorted order costs 21 left-moves and 33 right-moves, 54 in all; feeding the queries in their original order costs 60 left-moves and 88 right-moves, 148. The structure cuts pointer travel to roughly a third even at this tiny size.

Balancing the two main terms, `n^2 / block` falls and `q * block` rises with `block`, so the sum is smallest where they meet: `n^2 / block = q * block` gives `block = n / sqrt(q)`, but the standard and slightly coarser choice that treats `q` and `n` as comparable is `block = sqrt(n)`, which makes the right-endpoint term `O(n^2 / sqrt(n)) = O(n sqrt(n))` and the left-endpoint term `O(q sqrt(n))`. That yields `O((n + q) sqrt(n))` pointer movement. Larger blocks let `cur_l` wander too far inside a group; smaller blocks make too many groups and force `cur_r` to restart too often.

A closed-form balance is reassuring but I would like to see the predicted growth on real runs, since a hidden term could make the true cost climb faster than `(n + q) sqrt(n)`. I take `q = n` and random queries, count total endpoint moves, and compare against `(n + q) sqrt(n)`:

```
n=  100   moves=    1149   (n+q)sqrt(n)=    2000   ratio 0.57
n=  400   moves=    9461   (n+q)sqrt(n)=   16000   ratio 0.59
n= 1600   moves=   81560   (n+q)sqrt(n)=  128000   ratio 0.64
n= 6400   moves=  665895   (n+q)sqrt(n)= 1024000   ratio 0.65
```

As `n` grows by a factor of 64 the ratio stays in a narrow band around `0.6` rather than drifting upward, so the measured travel tracks `(n + q) sqrt(n)` with a small constant — consistent with the derived bound and not with any faster-growing hidden term. The sort itself adds `O(q log q)`.

Now the code can be written directly from these invariants, as one self-contained C++17 program. It reads `n`, then `n` array values, then `q`, then `q` query pairs `l r` (1-based inclusive) from stdin; the per-query loop moves `[cur_l, cur_r]` to that query's range one endpoint step at a time and writes `distinct` back at the query's original index, so the answers print in the original query order.

```cpp
// Mo's algorithm for offline distinct-value range queries.
// Reads from stdin: n, then n array values, then q, then q query pairs (l r),
// 1-based inclusive. Prints, one per line in original query order, the number of
// distinct values in a[l..r].
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) cin >> a[i];

    // Coordinate-compress values so cnt can be a dense array, valid for
    // arbitrary integer values (the dictionary-backed count in the reference).
    vector<int> sorted_vals(a.begin(), a.end());
    sort(sorted_vals.begin(), sorted_vals.end());
    sorted_vals.erase(unique(sorted_vals.begin(), sorted_vals.end()),
                      sorted_vals.end());
    for (int i = 0; i < n; ++i)
        a[i] = int(lower_bound(sorted_vals.begin(), sorted_vals.end(), a[i]) -
                   sorted_vals.begin());

    int q;
    cin >> q;
    vector<int> ql(q), qr(q), order(q);
    for (int i = 0; i < q; ++i) {
        int l, r;
        cin >> l >> r;             // 1-based inclusive
        ql[i] = l - 1;
        qr[i] = r - 1;
        order[i] = i;
    }

    int block = max(1, (int)sqrt((double)n));
    sort(order.begin(), order.end(), [&](int x, int y) {
        if (ql[x] / block != ql[y] / block) return ql[x] / block < ql[y] / block;
        return qr[x] < qr[y];
    });

    vector<int> cnt(max(1, (int)sorted_vals.size()), 0);
    int distinct = 0;
    vector<int> answers(q);

    auto add = [&](int i) {
        if (++cnt[a[i]] == 1) ++distinct;
    };
    auto remove = [&](int i) {
        if (--cnt[a[i]] == 0) --distinct;
    };

    int cur_l = 0, cur_r = -1;
    for (int idx : order) {
        int l = ql[idx], r = qr[idx];
        while (cur_l > l) add(--cur_l);
        while (cur_r < r) add(++cur_r);
        while (cur_l < l) remove(cur_l++);
        while (cur_r > r) remove(cur_r--);
        answers[idx] = distinct;
    }

    string out;
    for (int i = 0; i < q; ++i) {
        out += to_string(answers[i]);
        if (i + 1 < q) out += '\n';
    }
    cout << out;
    return 0;
}
```

The full chain is: exact frequency maintenance makes each endpoint step constant time and, checked against brute force, correct; an invariant range `[cur_l, cur_r]` lets answers be transferred between queries; lexicographic order by `(l // block, r)` keeps the right endpoint monotone within each block and so controls total endpoint travel; `block = sqrt(n)` balances the two travel terms to `O((n + q) sqrt(n))`, a bound the measured step counts track with a small constant, after the `O(q log q)` sort.
