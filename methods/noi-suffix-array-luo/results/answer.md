# Suffix Array (prefix-doubling + radix sort) and the Height/LCP array

## Problem

For a string $S$ of length $n$, build the **suffix array** $SA$ — the
permutation of $1..n$ that lists the start positions of all suffixes
$\mathrm{Suffix}(i) = S[i..n]$ in lexicographic order — its inverse the **rank
array** ($\mathrm{Rank}[SA[i]] = i$), and the **height (LCP) array**, with
$\mathrm{height}[1]=0$ and, for $2 \le i \le n$,
$\mathrm{height}[i] = \mathrm{lcp}(\mathrm{Suffix}(SA[i-1]), \mathrm{Suffix}(SA[i]))$.

## Key idea

**Construction by prefix-doubling.** Don't compare suffixes as opaque strings
(each comparison can scan $O(n)$ characters). Instead rank every suffix by its
first $\ell$ characters and double $\ell$ each round. The first $2\ell$
characters of $\mathrm{Suffix}(i)$ are its first $\ell$ characters followed by the
first $\ell$ characters of $\mathrm{Suffix}(i+\ell)$, so its length-$2\ell$ sort
key is the pair of length-$\ell$ ranks

$$\big(\mathrm{rank}_\ell(i),\ \mathrm{rank}_\ell(i+\ell)\big),$$

both of which are known from the previous round. Sorting suffixes by their first
$2\ell$ characters is sorting these pairs lexicographically. After
$\lceil\log_2 n\rceil$ doublings, $\ell \ge n$ and the ranks are the final order.

**Radix sort makes each round linear.** The pair coordinates are integers in
$[0, n)$, so each round sorts with two stable counting sorts (each $O(n)$): by the
second coordinate, then by the first. Two refinements:

- *Second key for free.* The second coordinate of suffix $i$ is
  $\mathrm{rank}_\ell(i+\ell)$. The previous round left $SA$ sorted by the first
  coordinate; scanning it in order and emitting $SA[j]-\ell$ (when
  $SA[j] \ge \ell$) lists the suffixes in increasing second-key order with no
  sort. Suffixes with $i+\ell$ past the end have the smallest (empty) second key
  and go first. So only the first-key counting sort is actually performed.
- *Sentinel and missing halves.* Append one character strictly smaller than all
  others and unique; then every suffix is strictly ordered (none is a prefix of
  another). In the guarded implementation below, a missing second half is treated
  as one synthetic key smaller than every real rank.

Recompute ranks by scanning the new $SA$: adjacent suffixes share a rank iff equal
first key **and** equal second key. Stop early once all $n$ ranks are distinct.

**Height array by $h[i] \ge h[i-1]-1$.** Index the adjacent-LCP array by start
position: let $h[i] = \mathrm{height}[\mathrm{Rank}[i]]$ be the LCP of
$\mathrm{Suffix}(i)$ with its predecessor in sorted order. If $\mathrm{Suffix}(k)$
is the predecessor of $\mathrm{Suffix}(i-1)$ with LCP $h[i-1] > 1$, write
$\mathrm{Suffix}(i-1) = cA D\dots$, $\mathrm{Suffix}(k) = cA B\dots$ with
$|cA| = h[i-1]$ and $B < D$. Dropping the leading $c$: $\mathrm{Suffix}(i) = AD\dots$
and $\mathrm{Suffix}(k+1) = AB\dots$ still agree on $A$ (length $h[i-1]-1$) and
$\mathrm{Suffix}(k+1) < \mathrm{Suffix}(i)$. Every suffix between
$\mathrm{Suffix}(k+1)$ and $\mathrm{Suffix}(i)$ in sorted order must stay in the
same length-$|A|$ prefix block, so the immediate predecessor of $\mathrm{Suffix}(i)$
shares at least $h[i-1]-1$ characters with it, giving $h[i] \ge h[i-1] - 1$.
Walking $i$ in start-position order and carrying $k = h[i-1]$ (resetting to
$\max(k-1,0)$, then extending by character comparison) builds the whole array in
$O(n)$ amortized: after the reset, $i+k$ never moves backward, so there are at
most $n$ successful extensions and at most $n$ one-step drops.

## Algorithm

1. Append a unique smallest sentinel; map characters to dense integer ranks.
2. Counting-sort by the single character ($\ell = 1$).
3. Repeat, doubling $\ell$: build the second-key order off the previous $SA$
   (one scan); stable counting sort by the first key; recompute ranks by
   adjacent-pair equality; tighten the alphabet to the rank count; stop when all
   ranks are distinct.
4. Build $\mathrm{height}$ in one position-order pass using $h[i] \ge h[i-1]-1$.

## Code

```python
def build_suffix_array(r, n, m):
    """sa[0..n-1] sorting the n suffixes of r (integer codes in [0, m),
    r[n-1] the unique smallest sentinel) lexicographically. O(n log n)."""
    sa = [0] * n
    x = list(r)            # working integer key per suffix; seed with the code
    y = [0] * n            # scratch
    ws = [0] * max(m, n)   # bounded-key tally

    # Initial order by one code.
    for i in range(m):
        ws[i] = 0
    for i in range(n):
        ws[x[i]] += 1
    for i in range(1, m):
        ws[i] += ws[i - 1]
    for i in range(n - 1, -1, -1):
        ws[x[i]] -= 1
        sa[ws[x[i]]] = i

    j, p = 1, 1
    while p < n:
        # Build the order by the later half, then stably sort by the earlier half.
        p = 0
        for i in range(max(n - j, 0), n):
            y[p] = i
            p += 1
        for i in range(n):
            if sa[i] >= j:
                y[p] = sa[i] - j
                p += 1

        wv = [x[y[i]] for i in range(n)]
        for i in range(m):
            ws[i] = 0
        for i in range(n):
            ws[wv[i]] += 1
        for i in range(1, m):
            ws[i] += ws[i - 1]
        for i in range(n - 1, -1, -1):
            ws[wv[i]] -= 1
            sa[ws[wv[i]]] = y[i]

        # Reuse the arrays: y is the previous key array, x receives new keys.
        x, y = y, x
        p = 1
        x[sa[0]] = 0
        for i in range(1, n):
            a, b = sa[i - 1], sa[i]
            left_a = y[a + j] if a + j < n else -1
            left_b = y[b + j] if b + j < n else -1
            same = y[a] == y[b] and left_a == left_b
            if same:
                x[b] = p - 1
            else:
                x[b] = p
                p += 1

        m = p
        j *= 2
    return sa


def build_height(s, sa):
    """height[i] = LCP(Suffix(sa[i-1]), Suffix(sa[i])); height[0] = 0. O(n)."""
    n = len(s)
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i    # inverse of sa
    height = [0] * n

    k = 0
    for i in range(n):
        pos = rank[i]
        if pos == 0:
            k = 0
            continue
        if k:
            k -= 1
        j = sa[pos - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        height[pos] = k
    return height
```

## Complexity

- **Construction:** $O(n \log n)$ time — $O(\log n)$ doubling rounds, each one
  stable counting sort in $O(n)$ (the second key needs no sort). $O(n)$ memory
  (a constant number of length-$n$ integer arrays).
- **Height array:** $O(n)$ time, $O(n)$ memory, one position-order pass.
- With the sorted $SA$ and the adjacent-pair $\mathrm{height}$ array, the LCP of
  any two suffixes equals the minimum of $\mathrm{height}$ over the stretch of the
  sorted order between them — a range-minimum query.
