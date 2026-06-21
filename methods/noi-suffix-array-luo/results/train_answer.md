The task is to take a string $S$ of length $n$ — with $n$ possibly in the millions — and produce the suffix array $SA$, the permutation that lists the start positions of all $n$ suffixes $\mathrm{Suffix}(i) = S[i..n]$ in lexicographic order, together with its inverse the rank array ($\mathrm{rank}[SA[i]] = i$) and the adjacent longest-common-prefix array $\mathrm{height}$, where $\mathrm{height}[0]=0$ and $\mathrm{height}[i]$ is the LCP of the two suffixes at $SA[i-1]$ and $SA[i]$. The honest first attempt — materialize the $n$ suffixes and hand them to a comparison sort — is correct but fatal at scale. A comparison sort does $O(n\log n)$ comparisons, but a single comparison of two suffixes scans them character by character until they differ, and two suffixes can agree for a very long time: on $S = a^n$, $\mathrm{Suffix}(1)$ and $\mathrm{Suffix}(2)$ share $n-1$ characters. So one comparison is $O(n)$ in the worst case, the whole sort is $O(n^2 \log n)$, and merely storing the suffixes is $O(n^2)$ memory. The deeper waste is that the character-scanning work done to compare one pair tells me a great deal about the next pair, and I throw it away each time. The cure is to stop treating a suffix as an opaque string to be re-scanned and instead rank every suffix once, then read order off the ranks in constant time.

I propose building the suffix array by prefix-doubling with radix sort, and building the height array by the position-indexed recurrence $h[i] \ge h[i-1]-1$. The central observation is this: instead of sorting suffixes by all their characters, sort them by their first $\ell$ characters, giving each suffix a rank $\mathrm{rank}_\ell(i)$ with ties (equal length-$\ell$ prefixes) sharing a rank, and then double $\ell$ cheaply by reusing the level below. Suppose I already have $\mathrm{rank}_\ell$ for every suffix and want $\mathrm{rank}_{2\ell}$. The first $2\ell$ characters of $\mathrm{Suffix}(i)$ split exactly in half: characters $S[i..i+\ell)$ then $S[i+\ell..i+2\ell)$. The first half is the length-$\ell$ prefix of $\mathrm{Suffix}(i)$, whose rank I have; the second half is the length-$\ell$ prefix of $\mathrm{Suffix}(i+\ell)$, which is also a suffix, so I have its rank too. Therefore the length-$2\ell$ prefix of suffix $i$ is captured, for all sorting purposes, by the pair

$$\big(\mathrm{rank}_\ell(i),\ \mathrm{rank}_\ell(i+\ell)\big),$$

and two suffixes compare on their first $2\ell$ characters exactly as these pairs compare lexicographically — first coordinate decides, second breaks ties. So one doubling round is just sorting $n$ pairs built entirely from the previous round's ranks. Starting at $\ell = 1$, where $\mathrm{rank}_1(i)$ is the order of the single character $S[i]$, and doubling $1 \to 2 \to 4 \to \dots$, after $\lceil\log_2 n\rceil$ rounds $\ell \ge n$, every length-$\ell$ prefix is the whole suffix, and the ranks are the final order. That is $O(\log n)$ rounds; the whole game is to make a round cost $O(n)$.

What makes a round linear is that the pair coordinates are not arbitrary keys but integer ranks in $[0, n)$, so I sort with counting sort rather than comparisons. Counting sort tallies how many items have each key, takes prefix sums to locate each key's block, and places the items in $O(n + m)$; walking the input right-to-left makes it stable, so equal-keyed items keep their input order. A pair needs two passes of stable counting sort, least-significant key first. The order is the part that is easy to get backwards, so I reason it rather than recite it: I want the first coordinate primary and the second to break ties, so I sort by the second coordinate first, then stably by the first; stability means that among items sharing a first coordinate the already-established second-coordinate order survives, yielding exactly pair-lexicographic order. Doing it the other way would scramble the first-coordinate grouping. Now the key refinement that removes a whole sort per round: the second coordinate of suffix $i$ is $\mathrm{rank}_\ell(i+\ell)$, and the previous round already left $SA$ sorted by the first coordinate. Scanning that $SA$ in increasing order, $\mathrm{rank}_\ell(SA[j])$ is increasing by construction, and the suffix whose second key equals $\mathrm{rank}_\ell(SA[j])$ is the one starting $\ell$ positions earlier, at $SA[j]-\ell$. So emitting $SA[j]-\ell$ whenever it is a valid position ($SA[j] \ge \ell$) lists the suffixes in increasing second-key order with no sorting at all — the previous $SA$, read with a shift, *is* the second-key order. The suffixes this scan misses are exactly those with $i+\ell \ge n$, whose second half is empty and therefore smallest; I prepend them (positions $n-\ell, \dots, n-1$). Only the first-key counting sort is then actually performed.

Two more design choices make the bookkeeping clean. First, I append to $S$ one sentinel character strictly smaller than every real character and appearing nowhere else. This does two things: no suffix is a prefix of another any longer, since at the position where a shorter suffix would end it now carries the unique smallest sentinel, so every pair of suffixes is strictly ordered and there are no full-length ties; and the sentinel, being the unique smallest suffix, lands at $SA[0]$ from the first round and stays. A missing second half (when $i+\ell$ runs past the end) is then handled in the guarded implementation as one synthetic key smaller than every real rank — exactly one representation of "empty second half is smallest." Second, after the stable first-key sort I recompute ranks by scanning the new $SA$: adjacent suffixes $SA[i-1]$ and $SA[i]$ share a new rank iff their length-$2\ell$ prefixes are identical, which, since the prefix is captured by the pair, means equal first coordinate *and* equal second coordinate — a constant-time check against the old ranks. Assign $SA[0]$ rank $0$ and increment whenever the pair changes; the largest rank $p$ is the number of distinct length-$2\ell$ prefixes. This gives a free early exit: the moment $p$ reaches $n$ every suffix has a distinct rank, the order can no longer change, and I stop — which on typical strings happens after very few rounds. It also lets me tighten the counting-sort alphabet bound to $m = p$ for the next round, keeping each pass at $O(n+p)=O(n)$. To avoid copying $n$ integers each round, I keep two buffers $x$ and $y$ and swap references so the old rank array becomes scratch and the new ranks are written in place. Each round is thus one linear scan plus one $O(n)$ counting sort, over $O(\log n)$ rounds, for $O(n\log n)$ total in $O(n)$ memory.

With $SA$ and $\mathrm{rank}$ in hand, the height array is the auxiliary object that makes the sorted suffixes useful: every pairwise LCP of two suffixes equals the minimum of the adjacent $\mathrm{height}$ values across the stretch of sorted order between them, because any suffix lying between two strings in sorted order must begin with their common prefix. The naive fill — compare each adjacent pair from the front — is again $O(n^2)$ re-scanning. The fix is to change the axis along which I chain the work. Adjacent entries of $\mathrm{height}$ are adjacent in *sorted* order, $SA[i]$ versus $SA[i+1]$, between which there is no string relationship. So instead I walk suffixes in *position* order and define $h[i] = \mathrm{height}[\mathrm{rank}[i]]$, the LCP of $\mathrm{Suffix}(i)$ with whatever ranks immediately before it. The same numbers, indexed by start position — and now position $i$ and position $i-1$ differ by chopping one leading character, since $\mathrm{Suffix}(i)$ is $\mathrm{Suffix}(i-1)$ with its head removed, which is a relationship I can exploit. Let $\mathrm{Suffix}(k)$ be the predecessor of $\mathrm{Suffix}(i-1)$, so their LCP is $h[i-1]$; if $h[i-1] \le 1$ the bound below is trivial, so take $h[i-1]>1$ and write their shared prefix as a leading $c$ followed by a block $A$ of length $h[i-1]-1$, with $\mathrm{Suffix}(k)=cA B\dots$ and $\mathrm{Suffix}(i-1)=cA D\dots$ differing after $cA$ with $B < D$ since $k$ ranks earlier. Drop the leading $c$ from both: $\mathrm{Suffix}(i-1)$ becomes $\mathrm{Suffix}(i)=AD\dots$ and $\mathrm{Suffix}(k)$ becomes $\mathrm{Suffix}(k+1)=AB\dots$, still agreeing on all of $A$ and still splitting with $B < D$, so $\mathrm{Suffix}(k+1) < \mathrm{Suffix}(i)$ with LCP exactly $|A| = h[i-1]-1$. Every suffix between $\mathrm{Suffix}(k+1)$ and $\mathrm{Suffix}(i)$ in sorted order stays inside the block beginning with that length-$|A|$ prefix, so the immediate predecessor of $\mathrm{Suffix}(i)$ shares at least those characters, giving

$$h[i] \ge h[i-1] - 1.$$

That inequality is the lever. Walking $i$ in position order, I carry a running $k$ equal to the previous LCP; at each step I reset it to $\max(k-1,0)$ rather than to zero, locate the predecessor $\mathrm{Suffix}(SA[\mathrm{rank}[i]-1])$, and extend by character comparison from offset $k$, incrementing while characters match; the final $k$ is $h[i]=\mathrm{height}[\mathrm{rank}[i]]$ (and when $\mathrm{rank}[i]=0$ there is no predecessor, so $\mathrm{height}[0]=0$ and $k$ resets). This is $O(n)$ amortized: after the at-most-one-step drop, the examined text position $i+k$ never moves backward as $i$ advances and each successful comparison advances it by one, so there are at most $n$ successful extensions and at most $n$ drops. Together, doubling with radix sort yields $SA$ and $\mathrm{rank}$ in $O(n\log n)$, and this single position-order pass yields $\mathrm{height}$ in $O(n)$ on top.

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
