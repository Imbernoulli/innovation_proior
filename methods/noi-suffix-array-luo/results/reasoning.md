The thing in front of me is a string $S$ of length $n$ — think $n$ in the millions — and I want the $n$ suffixes sorted into dictionary order, given as the array $SA$ of their start positions. The completely honest first move: make the $n$ suffixes and sort them. Generate $\mathrm{Suffix}(1), \dots, \mathrm{Suffix}(n)$, hand them to a comparison sort, read off the start positions. Correct, and on a small string fine. So let me cost it, because I already suspect it dies at $n = 10^6$.

A comparison sort does $O(n\log n)$ comparisons. The trap is in one comparison. Comparing two suffixes means scanning them character by character until they differ — and two suffixes can agree for a very long time. On $S = a^n$, $\mathrm{Suffix}(1)$ and $\mathrm{Suffix}(2)$ agree on $n-1$ characters before the shorter one runs out. So a single comparison is $O(n)$ in the worst case, and the whole sort is $O(n^2 \log n)$. And if I literally materialize the suffixes I've already spent $O(n^2)$ memory holding substrings of a string I already have. At a million characters both are hopeless. So the suffix-as-an-opaque-string-to-be-compared framing is the enemy; I'm paying $O(n)$ per comparison and throwing the work away each time.

Stare at *why* it's wasteful. When I compare $\mathrm{Suffix}(1)$ and $\mathrm{Suffix}(2)$ on $a^n$ I scan a huge shared prefix. Then I compare $\mathrm{Suffix}(1)$ and $\mathrm{Suffix}(3)$ and scan almost the same shared prefix again. The character-comparison work I did for one pair tells me a lot about the next pair, and I keep it for nothing. So I don't want to compare suffixes by re-scanning characters at all. I want to *rank* them once and then read order off the ranks in constant time.

Suppose, instead of fully sorting the suffixes, I only sort them by their first $\ell$ characters — assign each suffix a rank according to where its length-$\ell$ prefix falls in dictionary order, with ties (suffixes whose first $\ell$ characters are identical) getting equal rank. Call that $\mathrm{rank}_\ell(i)$. If I had $\mathrm{rank}_\ell$ for an $\ell$ big enough that no two suffixes share their first $\ell$ characters, I'd be done: distinct ranks for all suffixes means I've sorted them. The worst case forces $\ell$ up to $n$ (again $a^n$: the suffixes only separate at the very end), so I can't just pick a small $\ell$. But maybe I can climb to large $\ell$ cheaply if each level reuses the one below it.

So suppose I already have $\mathrm{rank}_\ell$ for *every* suffix, and I want $\mathrm{rank}_{2\ell}$. The first $2\ell$ characters of $\mathrm{Suffix}(i)$ split exactly down the middle: characters $S[i..i+\ell)$ followed by characters $S[i+\ell..i+2\ell)$. The first half is the length-$\ell$ prefix of $\mathrm{Suffix}(i)$ — and I have its rank, $\mathrm{rank}_\ell(i)$. The second half is the length-$\ell$ prefix of $\mathrm{Suffix}(i+\ell)$ — and *that's* also a suffix, so I have its rank too, $\mathrm{rank}_\ell(i+\ell)$. So the length-$2\ell$ prefix of suffix $i$ is completely described, for sorting purposes, by the pair

$$\big(\mathrm{rank}_\ell(i),\ \mathrm{rank}_\ell(i+\ell)\big).$$

Two suffixes compare on their first $2\ell$ characters exactly the way these pairs compare: first coordinate decides; on a tie, the second coordinate decides. That's just lexicographic order on the pair. So sorting all suffixes by their first $2\ell$ characters is sorting these $n$ pairs — and the pairs are built entirely from $\mathrm{rank}_\ell$, which I already have. Each level reuses the level below. Start at $\ell = 1$: $\mathrm{rank}_1(i)$ is just the order of the single character $S[i]$, trivially obtained by sorting the characters. Then double: $1 \to 2 \to 4 \to 8 \to \dots$. After $\lceil \log_2 n\rceil$ doublings, $\ell \ge n$, every length-$\ell$ prefix is the whole suffix, and the ranks are the final order. So this is at most about $\log n$ rounds. The question is the cost of one round.

One round sorts $n$ pairs and reassigns ranks. If I sort the pairs with a comparison sort, that's $O(n\log n)$ per round and $O(n\log^2 n)$ overall — better than $O(n^2\log n)$, but I have a sharper tool available, because the pair coordinates are not arbitrary: they're integer ranks, each in $[0, n)$. Sorting $n$ items whose keys are integers in a bounded range $[0, m)$ doesn't need comparisons at all — counting sort does it in $O(n + m)$: tally how many items have each key, take prefix sums to find where each key's block starts, then place the items. And crucially counting sort can be made *stable* — items with equal keys keep their input order — by walking the input from right to left and placing each item just before the running end of its key's block. With $m \le n$ that's $O(n)$ per pass.

But my key is a pair, two coordinates. Radix sort handles that: sort by the digits one at a time, least significant first, each pass a *stable* counting sort. With two coordinates that's two passes. The order matters and it's the part that's easy to get backwards, so let me reason it out rather than recite it. I want the final order to be: first coordinate primary, second coordinate breaks ties. If I sort by the *second* coordinate first, the items are now arranged by second coordinate. Then I do a *stable* sort by the first coordinate. Stability means: among items that share a first coordinate, the sort preserves the order they were already in — which is the second-coordinate order I just established. So after the second pass, items are grouped by first coordinate, and within each group they're in second-coordinate order. That's exactly pair-lexicographic. If I'd done first coordinate first and second coordinate second, the second pass would scramble the first-coordinate grouping. So: less significant key first, more significant key last — sort by second coordinate, then stably by first. Two $O(n)$ counting sorts per round, $\log n$ rounds, $O(n\log n)$ total. That's the target.

Now the bookkeeping inside a round, concretely. Say this round I have $x[i] = \mathrm{rank}_\ell(i)$ for every $i$, and I want the new $SA$ sorted by the first $2\ell$ characters, then the new ranks $\mathrm{rank}_{2\ell}$. First the sort-by-second-coordinate. The second coordinate of suffix $i$ is $\mathrm{rank}_\ell(i+\ell)$ — the rank of the suffix starting $\ell$ further along. Two awkward bits. The first: for suffixes near the end, $i + \ell \ge n$, there *is* no character there; the second half is empty. An empty second half should count as smaller than any real one, so those suffixes get the smallest possible second coordinate and sort first among the ties. Fine — I can handle them as a special smallest value.

The second awkward bit is more interesting, and it's where I can avoid a whole counting sort. I'm about to sort the suffixes by their second coordinate $\mathrm{rank}_\ell(i+\ell)$. But look at what the *previous* round handed me: $SA$, the suffixes already sorted by their first coordinate $\mathrm{rank}_\ell(\cdot)$. Walk that $SA$ from rank $0$ upward. As $j$ increases, $SA[j]$ is a suffix start position, and $\mathrm{rank}_\ell(SA[j])$ is increasing — that's what "sorted by first key" means. Now I want suffixes ordered by *their* second key. The suffix whose second key is $\mathrm{rank}_\ell(SA[j])$ is the one starting $\ell$ *before* $SA[j]$, namely position $SA[j] - \ell$ (when that's a real position, $SA[j] \ge \ell$). So as I scan $SA$ in increasing order and emit $SA[j] - \ell$ whenever it's valid, I'm emitting suffixes in increasing order of their second key — because I'm walking the second-key values in increasing order by construction. No sort needed: the previous round's $SA$ *is* the second-key order, read with a shift. The only suffixes this misses are exactly the ones with $i + \ell \ge n$ — empty second half, smallest second key — and I prepend those (positions $n-\ell, \dots, n-1$) at the front. So building the second-key order is one linear scan, and I've turned two counting sorts per round into one. The remaining pass — the stable counting sort by the first key $x[\cdot]$, fed with the items in this second-key order — is the only real sort.

Let me also handle a nuisance that's been lurking: those out-of-range second halves, and more generally the fact that suffixes have different lengths. Comparing a short suffix against a long one, dictionary order says the shorter is smaller if it's a prefix of the longer. I keep having to special-case "the second half ran off the end." There's a clean trick: append to $S$ one extra character that is strictly smaller than every real character and appears nowhere else — a sentinel, conceptually a $\$$ or a $0$. Now every suffix ends in this unique smallest character. Two consequences. First, no suffix is a prefix of another anymore: at the position where the shorter one would end, it now has the sentinel, which is smaller than whatever the longer one has there, so the comparison resolves cleanly and *every* pair of suffixes is strictly ordered — no ties at full length. Second, the code can treat any missing second half as one synthetic key smaller than every real rank, so "empty second half is smallest" has exactly one representation in the pair comparison. The sentinel is also the unique smallest suffix, so it lands at $SA[0]$ from the very first round and stays put.

After the stable first-key sort I have the new $SA$, sorted by the first $2\ell$ characters. Now I need the new ranks $\mathrm{rank}_{2\ell}$, which I'll write back into $x$. Walk the new $SA$ in order. Adjacent suffixes $SA[i-1]$ and $SA[i]$ get the same new rank exactly when their length-$2\ell$ prefixes are identical — which, since the prefix is captured by the pair, means same first coordinate *and* same second coordinate: $\mathrm{rank}_\ell(SA[i-1]) = \mathrm{rank}_\ell(SA[i])$ and $\mathrm{rank}_\ell(SA[i-1]+\ell) = \mathrm{rank}_\ell(SA[i]+\ell)$. I still have the old $\mathrm{rank}_\ell$ around (call it $y$ now — I'll get to the pointer shuffle), so this is a constant-time check per adjacent pair. Equal pair: same rank as the previous suffix. Different pair: rank increments. Assign $SA[0]$ rank $0$ and sweep. The largest rank assigned, call it $p$, is the number of *distinct* length-$2\ell$ prefixes. And here's a free early exit: if $p$ ever reaches $n$, every suffix already has a distinct rank — the suffixes are fully separated, further doubling can't change a thing, and I stop. (On a random string this kicks in after very few rounds; only adversarial near-periodic strings push it to the full $\log n$.) Also, the new ranks live in $[0, p)$, so I can shrink the counting-sort alphabet bound to $m = p$ for the next round — keeps each counting sort at $O(n + p) = O(n)$ tightly rather than $O(n + |\Sigma|)$.

One implementation nicety while I'm here: each round I produce a fresh rank array from the old one. Copying $n$ integers every round is wasteful when I can just keep two arrays $x, y$ and swap which is "old" and which is "new" by swapping references — a pointer swap, not an array copy. So I'll let $x$ hold current ranks, build the new ranks into the other buffer, swap, repeat.

Let me trace one full round on $S = \texttt{aabaaaab}$ with the sentinel appended (so $n = 9$, last position is the sentinel) to make sure I believe the pair logic. $\ell = 1$: rank by single character. The sentinel is smallest, then $\texttt{a}$, then $\texttt{b}$. So $\mathrm{rank}_1$ is $0$ for the sentinel position, the same small value for every $\texttt{a}$, the larger value for the two $\texttt{b}$s; $SA$ after this is the sentinel first, then the $\texttt{a}$ positions (tied, in some stable order), then the $\texttt{b}$ positions. Now $\ell = 1 \to 2$. The pair for suffix $i$ is $(\mathrm{rank}_1(i), \mathrm{rank}_1(i+1))$ — its first two characters as ranks. I build the second-key order by reading the current $SA$ and shifting back by $1$, prepend the one position whose second half is past the end, then stably counting-sort by the first key $\mathrm{rank}_1(i)$. Suffixes starting $\texttt{aa}$ now separate from those starting $\texttt{ab}$ from $\texttt{ba}$, and the $\texttt{a}$ block that was one big tie splits according to the second character. Doubling again to $\ell = 4$ separates by the first four characters, and one more reaches eight, at which point — checking — all nine ranks are distinct, $p = n$, and I stop. The final $SA$ for $\texttt{aabaaaab}\$$ comes out $[8, 3, 4, 5, 0, 6, 1, 7, 2]$, i.e. the sentinel, then the suffixes in dictionary order, which is what a brute-force sort of the suffixes gives. Good — the pair-doubling reproduces the true order.

So the construction is settled: counting-sort the characters for $\ell = 1$; then repeatedly double $\ell$, each round building the second-key order for free off the previous $SA$ by a shift, doing one stable counting sort by the first key, recomputing ranks by adjacent-pair equality, swapping buffers, and stopping the moment all ranks are distinct. Each round is $O(n)$, there are $O(\log n)$ rounds, so $O(n\log n)$, with a handful of length-$n$ integer arrays for memory.

Now the height array. With the suffixes sorted I want, for each adjacent pair in sorted order, the length of their longest common prefix — $\mathrm{height}[i] = \mathrm{lcp}\big(\mathrm{Suffix}(SA[i-1]), \mathrm{Suffix}(SA[i])\big)$. Why adjacent pairs and not all pairs? Because of a property of sorted order: take any two suffixes $j$ and $k$ with $\mathrm{rank}[j] < \mathrm{rank}[k]$, and let their common prefix have length $L$. Every suffix between them in sorted order must also begin with that same length-$L$ prefix; otherwise it would fall outside the interval between those two strings. So every adjacent pair across the interval has LCP at least $L$. At the same time, the first adjacent boundary where the shared prefix of the whole interval stops pins the value down, so the LCP of $j$ and $k$ is the minimum of the adjacent $\mathrm{height}$ values across that stretch. The adjacent-pair array $\mathrm{height}$ is the compressed object: every pairwise LCP is a range-minimum over it. That's why building $\mathrm{height}$ is worth a dedicated effort — it's the auxiliary array that makes the sorted suffixes actually useful.

The naive way to fill $\mathrm{height}$: for each adjacent pair, compare characters from the front until they differ. Each entry up to $O(n)$, total $O(n^2)$ — the same re-scanning waste I fought at the start, now on the LCP side. So I want a relationship between consecutive $\mathrm{height}$ entries that lets me reuse work. The trouble is that "consecutive entries of $\mathrm{height}$" are consecutive in *sorted* order, $SA[i-1]$ vs $SA[i]$ then $SA[i]$ vs $SA[i+1]$, and there's no obvious string relationship between $SA[i]$ and $SA[i+1]$ — they can start anywhere. Adjacent in sorted order is the wrong axis to chain along.

Let me change the axis. Instead of walking $\mathrm{height}$ in sorted order, walk the suffixes in *position* order $i = 1, 2, 3, \dots$, and define $h[i]$ = the LCP of $\mathrm{Suffix}(i)$ with whatever suffix sits immediately before it in sorted order; that is, $h[i] = \mathrm{height}[\mathrm{rank}[i]]$. Same numbers, indexed by start position instead of by rank. The reason to index by position: position $i$ and position $i-1$ differ by dropping a single leading character — $\mathrm{Suffix}(i)$ is $\mathrm{Suffix}(i-1)$ with its first character chopped off — and *that's* a string relationship I can exploit.

So suppose I know $h[i-1]$ and want a lower bound on $h[i]$. Let $\mathrm{Suffix}(k)$ be the suffix ranked immediately before $\mathrm{Suffix}(i-1)$; by definition their LCP is $h[i-1]$. If $h[i-1] \le 1$ there's nothing to prove — $h[i] \ge 0 \ge h[i-1]-1$ holds trivially, so assume $h[i-1] > 1$. Write the shared prefix of $\mathrm{Suffix}(k)$ and $\mathrm{Suffix}(i-1)$ as a leading character $c$ followed by a block $A$ of length $h[i-1]-1$. So $\mathrm{Suffix}(k) = c\,A\,B\dots$ and $\mathrm{Suffix}(i-1) = c\,A\,D\dots$, where after the shared $cA$ they differ, and since $k$ ranks before $i-1$, the differing tail of $k$ is the smaller: $B < D$ at the first differing character. Now chop the leading $c$ off both. $\mathrm{Suffix}(i-1)$ with its head removed is exactly $\mathrm{Suffix}(i)$, and it now reads $A\,D\dots$. $\mathrm{Suffix}(k)$ with its head removed is $\mathrm{Suffix}(k+1)$, reading $A\,B\dots$. The two still agree on all of $A$ and then split with $B < D$ — so $\mathrm{Suffix}(k+1) < \mathrm{Suffix}(i)$, and their LCP is exactly $|A| = h[i-1] - 1$. So here is a suffix, $\mathrm{Suffix}(k+1)$, that ranks somewhere before $\mathrm{Suffix}(i)$ and shares a prefix of length $h[i-1]-1$ with it. In sorted order, every string between $\mathrm{Suffix}(k+1)$ and $\mathrm{Suffix}(i)$ must stay inside the block of strings beginning with that same length-$|A|$ prefix, so the suffix ranked immediately before $\mathrm{Suffix}(i)$ shares at least those $|A|$ characters too. That gives

$$h[i] \ge h[i-1] - 1.$$

That's the lever. Walk $i$ from $1$ upward carrying a running $k$ that holds the current LCP length. The inequality says when I step from $i-1$ to $i$, the new LCP is at least the old one minus one — so I don't restart the character comparison from zero, I start it from $\max(k-1, 0)$. To extend it I find the suffix ranked just before $\mathrm{Suffix}(i)$, namely $\mathrm{Suffix}(SA[\mathrm{rank}[i]-1])$, and compare characters from offset $k$ onward, incrementing $k$ while they match; the final $k$ is $h[i] = \mathrm{height}[\mathrm{rank}[i]]$. (If $\mathrm{rank}[i] = 0$, $\mathrm{Suffix}(i)$ is the smallest suffix and has no predecessor, so $\mathrm{height}[0] = 0$ and I reset $k$.) Why this is linear: after the possible one-step drop, the examined text position $i+k$ never moves backward as $i$ advances, and every successful comparison increments it by one. That gives at most $n$ successful extensions, while the one-step drops also happen at most $n$ times. The matching work is bounded by those extensions plus those drops, both $O(n)$ — so the entire height array is built in $O(n)$, amortized, with a single pass.

Putting the two together: doubling with radix sort gives $SA$ and $\mathrm{rank}$ in $O(n\log n)$, and this position-order recurrence gives $\mathrm{height}$ in $O(n)$ on top. Let me write it. I'll keep the construction faithful to the tight integer-array form: $x$ holds current ranks, $y$ doubles as the second-key order and then the old ranks, $sa$ the permutation, and a single counting array; the sentinel is an appended smallest character, while missing second halves are guarded explicitly as the synthetic smallest key.

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

The whole chain: comparing suffixes as opaque strings costs $O(n)$ per comparison because they share long prefixes and the work is thrown away, so instead I rank every suffix by a bounded-length prefix and double that length, since a length-$2\ell$ prefix is the pair of length-$\ell$ ranks of the suffix and the one $\ell$ further on; each doubling sorts those pairs in linear time with a stable counting sort — and the second key needs no sort at all because the previous round's $SA$, read with a shift, already lists the suffixes in second-key order — so the sort is $O(n\log n)$, an appended smallest sentinel makes every suffix strictly ordered, missing second halves compare as one synthetic smallest key, and ranks are recomputed by adjacent-pair equality with an early stop when all ranks separate; then, indexing the adjacent-LCP array by start position turns "drop one leading character" into the inequality $h[i] \ge h[i-1]-1$, which lets a single position-order pass build the height array in $O(n)$.
