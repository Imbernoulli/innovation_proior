The thing in front of me is a string $S$ of length $n$ — think $n$ in the millions — and I want the $n$ suffixes sorted into dictionary order, given as the array $SA$ of their start positions. The completely honest first move: make the $n$ suffixes and sort them. Generate $\mathrm{Suffix}(1), \dots, \mathrm{Suffix}(n)$, hand them to a comparison sort, read off the start positions. Correct, and on a small string fine. So let me cost it, because I already suspect it dies at $n = 10^6$.

A comparison sort does $O(n\log n)$ comparisons. The trap is in one comparison. Comparing two suffixes means scanning them character by character until they differ — and two suffixes can agree for a very long time. On $S = a^n$, $\mathrm{Suffix}(1)$ and $\mathrm{Suffix}(2)$ agree on $n-1$ characters before the shorter one runs out. So a single comparison is $O(n)$ in the worst case, and the whole sort is $O(n^2 \log n)$. And if I literally materialize the suffixes I've already spent $O(n^2)$ memory holding substrings of a string I already have. At a million characters both are hopeless. So the suffix-as-an-opaque-string-to-be-compared framing is the enemy; I'm paying $O(n)$ per comparison and throwing the work away each time.

Stare at *why* it's wasteful. When I compare $\mathrm{Suffix}(1)$ and $\mathrm{Suffix}(2)$ on $a^n$ I scan a huge shared prefix. Then I compare $\mathrm{Suffix}(1)$ and $\mathrm{Suffix}(3)$ and scan almost the same shared prefix again. The character-comparison work I did for one pair tells me a lot about the next pair, and I keep it for nothing. So I don't want to compare suffixes by re-scanning characters at all. I want to *rank* them once and then read order off the ranks in constant time.

Suppose, instead of fully sorting the suffixes, I only sort them by their first $\ell$ characters — assign each suffix a rank according to where its length-$\ell$ prefix falls in dictionary order, with ties (suffixes whose first $\ell$ characters are identical) getting equal rank. Call that $\mathrm{rank}_\ell(i)$. If I had $\mathrm{rank}_\ell$ for an $\ell$ big enough that no two suffixes share their first $\ell$ characters, I'd be done: distinct ranks for all suffixes means I've sorted them. The worst case forces $\ell$ up to $n$ (again $a^n$: the suffixes only separate at the very end), so I can't just pick a small $\ell$. But maybe I can climb to large $\ell$ cheaply if each level reuses the one below it.

So suppose I already have $\mathrm{rank}_\ell$ for *every* suffix, and I want $\mathrm{rank}_{2\ell}$. The first $2\ell$ characters of $\mathrm{Suffix}(i)$ split exactly down the middle: characters $S[i..i+\ell)$ followed by characters $S[i+\ell..i+2\ell)$. The first half is the length-$\ell$ prefix of $\mathrm{Suffix}(i)$ — and I have its rank, $\mathrm{rank}_\ell(i)$. The second half is the length-$\ell$ prefix of $\mathrm{Suffix}(i+\ell)$ — and *that's* also a suffix, so I have its rank too, $\mathrm{rank}_\ell(i+\ell)$. So the length-$2\ell$ prefix of suffix $i$ is completely described, for sorting purposes, by the pair

$$\big(\mathrm{rank}_\ell(i),\ \mathrm{rank}_\ell(i+\ell)\big).$$

Two suffixes compare on their first $2\ell$ characters exactly the way these pairs compare: first coordinate decides; on a tie, the second coordinate decides. That's just lexicographic order on the pair. So sorting all suffixes by their first $2\ell$ characters is sorting these $n$ pairs — and the pairs are built entirely from $\mathrm{rank}_\ell$, which I already have. Each level reuses the level below. Start at $\ell = 1$: $\mathrm{rank}_1(i)$ is just the order of the single character $S[i]$, trivially obtained by sorting the characters. Then double: $1 \to 2 \to 4 \to 8 \to \dots$. After $\lceil \log_2 n\rceil$ doublings, $\ell \ge n$, every length-$\ell$ prefix is the whole suffix, and the ranks are the final order. So this is at most about $\log n$ rounds. The question is the cost of one round.

One round sorts $n$ pairs and reassigns ranks. If I sort the pairs with a comparison sort, that's $O(n\log n)$ per round and $O(n\log^2 n)$ overall — better than $O(n^2\log n)$, but I have a sharper tool available, because the pair coordinates are not arbitrary: they're integer ranks, each in $[0, n)$. Sorting $n$ items whose keys are integers in a bounded range $[0, m)$ doesn't need comparisons at all — counting sort does it in $O(n + m)$: tally how many items have each key, take prefix sums to find where each key's block starts, then place the items. And crucially counting sort can be made *stable* — items with equal keys keep their input order — by walking the input from right to left and placing each item just before the running end of its key's block. With $m \le n$ that's $O(n)$ per pass.

But my key is a pair, two coordinates. Radix sort handles that: sort by the digits one at a time, least significant first, each pass a *stable* counting sort. With two coordinates that's two passes. The order matters and it's the part that's easy to get backwards, so let me reason it out rather than recite it. I want the final order to be: first coordinate primary, second coordinate breaks ties. If I sort by the *second* coordinate first, the items are now arranged by second coordinate. Then I do a *stable* sort by the first coordinate. Stability means: among items that share a first coordinate, the sort preserves the order they were already in — which is the second-coordinate order I just established. So after the second pass, items are grouped by first coordinate, and within each group they're in second-coordinate order. That's exactly pair-lexicographic. If I'd done first coordinate first and second coordinate second, the second pass would scramble the first-coordinate grouping.

Let me sanity-check that "less significant key first" claim on a tiny set of pairs, because it's the one place I'd believe a confident wrong answer. Take pairs $(1,1),(0,1),(1,0),(0,0)$ in that input order; lexicographic order should be $(0,0),(0,1),(1,0),(1,1)$. Sort stably by the *second* coordinate: the second coords are $1,1,0,0$, so the zeros come first keeping input order — $(1,0),(0,0),(0,1),(1,1)$. Now sort stably by the *first* coordinate: first coords are $1,0,0,1$; the zeros come first preserving their current order $(0,0),(0,1)$, then the ones $(1,0),(1,1)$ — giving $(0,0),(0,1),(1,0),(1,1)$. That matches. And to confirm the order can't be flipped: sort by *first* coordinate first, getting $(0,1),(0,0),(1,1),(1,0)$, then stably by second — the items with second coord $0$ are $(0,0),(1,0)$ in that order, then second coord $1$ are $(0,1),(1,1)$, yielding $(0,0),(1,0),(0,1),(1,1)$, which is wrong: $(1,0)$ before $(0,1)$. So the order is forced — less significant key first, more significant key last: sort by second coordinate, then stably by first. Two $O(n)$ counting sorts per round, $\log n$ rounds, $O(n\log n)$ overall.

Now the bookkeeping inside a round, concretely. Say this round I have $x[i] = \mathrm{rank}_\ell(i)$ for every $i$, and I want the new $SA$ sorted by the first $2\ell$ characters, then the new ranks $\mathrm{rank}_{2\ell}$. First the sort-by-second-coordinate. The second coordinate of suffix $i$ is $\mathrm{rank}_\ell(i+\ell)$ — the rank of the suffix starting $\ell$ further along. Two awkward bits. The first: for suffixes near the end, $i + \ell \ge n$, there *is* no character there; the second half is empty. An empty second half should count as smaller than any real one, so those suffixes get the smallest possible second coordinate and sort first among the ties. Fine — I can handle them as a special smallest value.

The second awkward bit is more interesting, and it's where I can avoid a whole counting sort. I'm about to sort the suffixes by their second coordinate $\mathrm{rank}_\ell(i+\ell)$. But look at what the *previous* round handed me: $SA$, the suffixes already sorted by their first coordinate $\mathrm{rank}_\ell(\cdot)$. Walk that $SA$ from rank $0$ upward. As $j$ increases, $SA[j]$ is a suffix start position, and $\mathrm{rank}_\ell(SA[j])$ is increasing — that's what "sorted by first key" means. Now I want suffixes ordered by *their* second key. The suffix whose second key is $\mathrm{rank}_\ell(SA[j])$ is the one starting $\ell$ *before* $SA[j]$, namely position $SA[j] - \ell$ (when that's a real position, $SA[j] \ge \ell$). So as I scan $SA$ in increasing order and emit $SA[j] - \ell$ whenever it's valid, I'm emitting suffixes in increasing order of their second key — because I'm walking the second-key values in increasing order by construction. No sort needed: the previous round's $SA$ *is* the second-key order, read with a shift. The only suffixes this misses are exactly the ones with $i + \ell \ge n$ — empty second half, smallest second key — and I prepend those (positions $n-\ell, \dots, n-1$) at the front. So building the second-key order is one linear scan, and I've turned two counting sorts per round into one. The remaining pass — the stable counting sort by the first key $x[\cdot]$, fed with the items in this second-key order — is the only real sort.

Let me also handle a nuisance that's been lurking: those out-of-range second halves, and more generally the fact that suffixes have different lengths. Comparing a short suffix against a long one, dictionary order says the shorter is smaller if it's a prefix of the longer. I keep having to special-case "the second half ran off the end." There's a clean trick: append to $S$ one extra character that is strictly smaller than every real character and appears nowhere else — a sentinel, conceptually a $\$$ or a $0$. Now every suffix ends in this unique smallest character. Two consequences. First, no suffix is a prefix of another anymore: at the position where the shorter one would end, it now has the sentinel, which is smaller than whatever the longer one has there, so the comparison resolves cleanly and *every* pair of suffixes is strictly ordered — no ties at full length. Second, the code can treat any missing second half as one synthetic key smaller than every real rank, so "empty second half is smallest" has exactly one representation in the pair comparison. The sentinel is also the unique smallest suffix, so it lands at $SA[0]$ from the very first round and stays put.

After the stable first-key sort I have the new $SA$, sorted by the first $2\ell$ characters. Now I need the new ranks $\mathrm{rank}_{2\ell}$, which I'll write back into $x$. Walk the new $SA$ in order. Adjacent suffixes $SA[i-1]$ and $SA[i]$ get the same new rank exactly when their length-$2\ell$ prefixes are identical — which, since the prefix is captured by the pair, means same first coordinate *and* same second coordinate: $\mathrm{rank}_\ell(SA[i-1]) = \mathrm{rank}_\ell(SA[i])$ and $\mathrm{rank}_\ell(SA[i-1]+\ell) = \mathrm{rank}_\ell(SA[i]+\ell)$. I still have the old $\mathrm{rank}_\ell$ around (call it $y$ now — I'll get to the pointer shuffle), so this is a constant-time check per adjacent pair. Equal pair: same rank as the previous suffix. Different pair: rank increments. Assign $SA[0]$ rank $0$ and sweep. The largest rank assigned, call it $p$, is the number of *distinct* length-$2\ell$ prefixes. And here's a free early exit: if $p$ ever reaches $n$, every suffix already has a distinct rank — the suffixes are fully separated, further doubling can't change a thing, and I stop. (On a random string this kicks in after very few rounds; only adversarial near-periodic strings push it to the full $\log n$.) Also, the new ranks live in $[0, p)$, so I can shrink the counting-sort alphabet bound to $m = p$ for the next round — keeps each counting sort at $O(n + p) = O(n)$ tightly rather than $O(n + |\Sigma|)$.

One implementation nicety while I'm here: each round I produce a fresh rank array from the old one. Copying $n$ integers every round is wasteful when I can just keep two arrays $x, y$ and swap which is "old" and which is "new" by swapping references — a pointer swap, not an array copy. So I'll let $x$ hold current ranks, build the new ranks into the other buffer, swap, repeat.

Let me trace the rounds on $S = \texttt{aabaaaab}$ with the sentinel appended, using $0$-based positions $0..8$ (so $n = 9$, position $8$ is the sentinel) — carrying the actual numbers, because I want to catch a bug in the pair logic, not narrate past it. Code the characters as sentinel $\to 0$, $\texttt{a} \to 1$, $\texttt{b} \to 2$, so the seed key array is $r = [1,1,2,1,1,1,1,2,0]$.

$\ell = 1$: counting-sort by the single code. The sentinel ($0$) is smallest, then the six $\texttt{a}$s ($1$), then the two $\texttt{b}$s ($2$). That gives $SA = [8,\ 0,1,3,4,5,6,\ 2,7]$ — position $8$ first, then the six $\texttt{a}$ positions in their stable input order $0,1,3,4,5,6$, then the two $\texttt{b}$ positions $2,7$. The rank array $x$ is still $[1,1,2,1,1,1,1,2,0]$.

$\ell = 1 \to 2$, $j = 1$. The pair for suffix $i$ is $(\mathrm{rank}_1(i),\ \mathrm{rank}_1(i+1))$. I build the second-key order $y$ by first listing positions with no real second half ($i + 1 \ge 9$, i.e. just position $8$), then scanning $SA$ and emitting $SA[k] - 1$ wherever $SA[k] \ge 1$: from $SA = [8,0,1,3,4,5,6,2,7]$ that emits $7,0,2,3,4,5,1,6$. So $y = [8,\ 7,0,2,3,4,5,1,6]$. Then I stably counting-sort these by their first key $x[\cdot]$. The result comes out $SA = [8,\ 0,3,4,5,\ 1,6,\ 7,2]$. Now recompute ranks by adjacent-pair equality. Reading down this $SA$, the pairs are: pos $8 \to (0,\cdot)$ giving rank $0$; pos $0 \to (1,1)$ rank $1$; pos $3 \to (1,1)$ — same pair, rank stays $1$; pos $4 \to (1,1)$ rank $1$; pos $5 \to (1,1)$ rank $1$; pos $1 \to (1,2)$ — first key same but second key jumps, new rank $2$; pos $6 \to (1,0)$… wait, let me actually use the synthetic-smallest convention for $i+j$ past the end and the real second keys, and just let the code's equality test decide. Running it, the new rank array lands as $x = [1,2,4,1,1,1,2,3,0]$ and the distinct-rank count is $p = 5$. Not yet $n$, so I keep going — and indeed positions $3,4,5$ still share rank $1$, because $\texttt{aa}$, $\texttt{aa}$, $\texttt{aa}$ are still tied on two characters. Good: the $\texttt{a}$ block split but didn't fully resolve, which is exactly what two characters of resolution should do.

$\ell = 2 \to 4$, $j = 2$. Same machinery with the length-$2$ ranks. The counting sort produces $SA = [8,\ 3,4,5,0,\ 6,1,\ 7,2]$, and recomputing ranks gives $x = [4,6,8,1,2,3,5,7,0]$ with $p = 9$. That hits $p = n$ — every suffix now has a distinct rank — so the early exit fires here and I stop. Note this is one round earlier than the worst case would need: I never had to double to $\ell = 8$, because four characters already separate all nine suffixes of this particular string. The final $SA$ is $[8, 3, 4, 5, 0, 6, 1, 7, 2]$.

Is that right? Let me check it against a literal sort of the suffixes (this is small enough to do directly). The suffixes in dictionary order are: $\texttt{\$}$ (pos $8$); $\texttt{aaaab\$}$ (pos $3$); $\texttt{aaab\$}$ (pos $4$); $\texttt{aab\$}$ (pos $5$); $\texttt{aabaaaab\$}$ (pos $0$); $\texttt{ab\$}$ (pos $6$); $\texttt{abaaaab\$}$ (pos $1$); $\texttt{b\$}$ (pos $7$); $\texttt{baaaab\$}$ (pos $2$). Reading off positions: $8,3,4,5,0,6,1,7,2$. That is exactly the $SA$ the doubling produced. The one spot worth double-checking is pos $3$ ($\texttt{aaaab\$}$) before pos $4$ ($\texttt{aaab\$}$): they agree on $\texttt{aaa}$, then pos $3$ has $\texttt{a}$ where pos $4$ has $\texttt{b}$, and $\texttt{a} < \texttt{b}$, so pos $3$ is smaller — matching the order. The pair-doubling reproduced the true order, and it did it in two rounds rather than the three a $\log_2 9$ bound would allow.

So the construction holds up on this example: counting-sort the characters for $\ell = 1$; then repeatedly double $\ell$, each round building the second-key order for free off the previous $SA$ by a shift, doing one stable counting sort by the first key, recomputing ranks by adjacent-pair equality, swapping buffers, and stopping the moment all ranks are distinct. Each round is $O(n)$, there are $O(\log n)$ rounds, so $O(n\log n)$, with a handful of length-$n$ integer arrays for memory.

Now the height array. With the suffixes sorted I want, for each adjacent pair in sorted order, the length of their longest common prefix — $\mathrm{height}[i] = \mathrm{lcp}\big(\mathrm{Suffix}(SA[i-1]), \mathrm{Suffix}(SA[i])\big)$. Why adjacent pairs and not all pairs? Because of a property of sorted order: take any two suffixes $j$ and $k$ with $\mathrm{rank}[j] < \mathrm{rank}[k]$, and let their common prefix have length $L$. Every suffix between them in sorted order must also begin with that same length-$L$ prefix; otherwise it would fall outside the interval between those two strings. So every adjacent pair across the interval has LCP at least $L$. At the same time, the first adjacent boundary where the shared prefix of the whole interval stops pins the value down, so the LCP of $j$ and $k$ is the minimum of the adjacent $\mathrm{height}$ values across that stretch. The adjacent-pair array $\mathrm{height}$ is the compressed object: every pairwise LCP is a range-minimum over it. That's why building $\mathrm{height}$ is worth a dedicated effort — it's the auxiliary array that makes the sorted suffixes actually useful.

The naive way to fill $\mathrm{height}$: for each adjacent pair, compare characters from the front until they differ. Each entry up to $O(n)$, total $O(n^2)$ — the same re-scanning waste I fought at the start, now on the LCP side. So I want a relationship between consecutive $\mathrm{height}$ entries that lets me reuse work. The trouble is that "consecutive entries of $\mathrm{height}$" are consecutive in *sorted* order, $SA[i-1]$ vs $SA[i]$ then $SA[i]$ vs $SA[i+1]$, and there's no obvious string relationship between $SA[i]$ and $SA[i+1]$ — they can start anywhere. Adjacent in sorted order is the wrong axis to chain along.

Let me change the axis. Instead of walking $\mathrm{height}$ in sorted order, walk the suffixes in *position* order $i = 1, 2, 3, \dots$, and define $h[i]$ = the LCP of $\mathrm{Suffix}(i)$ with whatever suffix sits immediately before it in sorted order; that is, $h[i] = \mathrm{height}[\mathrm{rank}[i]]$. Same numbers, indexed by start position instead of by rank. The reason to index by position: position $i$ and position $i-1$ differ by dropping a single leading character — $\mathrm{Suffix}(i)$ is $\mathrm{Suffix}(i-1)$ with its first character chopped off — and *that's* a string relationship I can exploit.

So suppose I know $h[i-1]$ and want a lower bound on $h[i]$. Let $\mathrm{Suffix}(k)$ be the suffix ranked immediately before $\mathrm{Suffix}(i-1)$; by definition their LCP is $h[i-1]$. If $h[i-1] \le 1$ there's nothing to prove — $h[i] \ge 0 \ge h[i-1]-1$ holds trivially, so assume $h[i-1] > 1$. Write the shared prefix of $\mathrm{Suffix}(k)$ and $\mathrm{Suffix}(i-1)$ as a leading character $c$ followed by a block $A$ of length $h[i-1]-1$. So $\mathrm{Suffix}(k) = c\,A\,B\dots$ and $\mathrm{Suffix}(i-1) = c\,A\,D\dots$, where after the shared $cA$ they differ, and since $k$ ranks before $i-1$, the differing tail of $k$ is the smaller: $B < D$ at the first differing character. Now chop the leading $c$ off both. $\mathrm{Suffix}(i-1)$ with its head removed is exactly $\mathrm{Suffix}(i)$, and it now reads $A\,D\dots$. $\mathrm{Suffix}(k)$ with its head removed is $\mathrm{Suffix}(k+1)$, reading $A\,B\dots$. The two still agree on all of $A$ and then split with $B < D$ — so $\mathrm{Suffix}(k+1) < \mathrm{Suffix}(i)$, and their LCP is exactly $|A| = h[i-1] - 1$. So here is a suffix, $\mathrm{Suffix}(k+1)$, that ranks somewhere before $\mathrm{Suffix}(i)$ and shares a prefix of length $h[i-1]-1$ with it. In sorted order, every string between $\mathrm{Suffix}(k+1)$ and $\mathrm{Suffix}(i)$ must stay inside the block of strings beginning with that same length-$|A|$ prefix, so the suffix ranked immediately before $\mathrm{Suffix}(i)$ shares at least those $|A|$ characters too. That gives

$$h[i] \ge h[i-1] - 1.$$

That's the lever. Walk $i$ from $1$ upward carrying a running $k$ that holds the current LCP length. The inequality says when I step from $i-1$ to $i$, the new LCP is at least the old one minus one — so I don't restart the character comparison from zero, I start it from $\max(k-1, 0)$. To extend it I find the suffix ranked just before $\mathrm{Suffix}(i)$, namely $\mathrm{Suffix}(SA[\mathrm{rank}[i]-1])$, and compare characters from offset $k$ onward, incrementing $k$ while they match; the final $k$ is $h[i] = \mathrm{height}[\mathrm{rank}[i]]$. (If $\mathrm{rank}[i] = 0$, $\mathrm{Suffix}(i)$ is the smallest suffix and has no predecessor, so $\mathrm{height}[0] = 0$ and I reset $k$.) Why this is linear: after the possible one-step drop, the examined text position $i+k$ never moves backward as $i$ advances, and every successful comparison increments it by one. That gives at most $n$ successful extensions, while the one-step drops also happen at most $n$ times. The matching work is bounded by those extensions plus those drops, both $O(n)$ — so the entire height array is built in $O(n)$, amortized, with a single pass.

Before I trust the $h[i] \ge h[i-1]-1$ argument I'd like to watch the carried $k$ actually behave, because the amortization claim is the whole point and it's exactly the kind of thing that's easy to believe and wrong. Take the same string with $SA = [8,3,4,5,0,6,1,7,2]$; inverting gives $\mathrm{rank}$ by position $= [4,6,8,1,2,3,5,7,0]$. Now walk positions $i = 0,1,2,3,\dots$ carrying $k$. At $i=0$ ($\mathrm{rank}=4$, predecessor pos $5 = \texttt{aab\$}$ vs $\texttt{aabaaaab\$}$): nothing carried, I extend and match $\texttt{aab}$, so $h = 3$. At $i=1$ ($\mathrm{rank}=6$): I drop $k$ to $2$ and extend — predecessor is pos $6 = \texttt{ab\$}$ vs $\texttt{abaaaab\$}$, which share $\texttt{ab}$, so $h$ stays $2$, no extension past the inherited $2$. At $i=2$ ($\mathrm{rank}=8$): drop to $1$, predecessor pos $7 = \texttt{b\$}$ vs $\texttt{baaaab\$}$ share $\texttt{b}$, $h = 1$. At $i=3$ ($\mathrm{rank}=1$): drop to $0$, predecessor pos $8 = \texttt{\$}$ vs $\texttt{aaaab\$}$ share nothing, $h = 0$. So across $i = 0,1,2,3$ the values run $3, 2, 1, 0$ — each exactly the previous minus one, the inequality biting at its tightest, and at every step I started the character comparison from the dropped $k$ rather than from zero. Then $i=4$ ($\mathrm{rank}=2$, predecessor pos $3$): $\mathrm{Suffix}(4) = \texttt{aaab\$}$ against $\mathrm{Suffix}(3) = \texttt{aaaab\$}$ jumps back up to $h=3$, extending $\texttt{aaa}$ from a dropped $k=0$ — a real climb, paid for by the text pointer $i+k$ advancing, never retreating. Indexed back by rank, these give $\mathrm{height} = [0,0,3,2,3,1,2,0,1]$ over ranks $0..8$. Checking a couple directly against the sorted list: rank $2$ is pos $4=\texttt{aaab\$}$ against rank $1$ pos $3=\texttt{aaaab\$}$ — they share $\texttt{aaa}$, LCP $3$, and $\mathrm{height}[2]=3$; rank $4$ is pos $0=\texttt{aabaaaab\$}$ against rank $3$ pos $5=\texttt{aab\$}$, sharing $\texttt{aab}$, LCP $3$, and $\mathrm{height}[4]=3$. Both match. The recurrence and its amortization do what the argument said.

Putting the two together: doubling with radix sort gives $SA$ and $\mathrm{rank}$ in $O(n\log n)$, and this position-order recurrence gives $\mathrm{height}$ in $O(n)$ on top. Let me write it as a single self-contained C++17 program that reads one line — the string $S$ — from standard input and prints $n$, the $sa[]$ row, then the $height[]$ row. I'll keep the construction faithful to the tight integer-array form: $x$ holds current ranks, $y$ doubles as the second-key order and then the old ranks, $sa$ the permutation, and a single counting array; the sentinel is an appended smallest character, while missing second halves are guarded explicitly as the synthetic smallest key. `main` does the boilerplate — append the sentinel, rank-compress characters to dense codes, call the two routines, print.

The shifted second-key order in the prefix-doubling radix construction is the part I'd most easily get wrong under pressure; if I weren't confident in it, I'd fall back to the plain two-pass stable counting-sort radix sort on $(\mathrm{rank}_\ell(i), \mathrm{rank}_\ell(i+\ell))$ each round and ship that correct variant.

```cpp
// Reads one line (the string S) from stdin. Appends a unique smallest sentinel,
// builds the suffix array sa[] (start positions of all suffixes, including the
// sentinel suffix, in lexicographic order) in O(n log n), then the height array
// (height[i] = LCP(Suffix(sa[i-1]), Suffix(sa[i])), height[0] = 0) in O(n).
// Prints n, then the sa[] row, then the height[] row (space-separated).
#include <bits/stdc++.h>
using namespace std;

// sa[0..n-1] sorts the n suffixes of r (integer codes in [0, m), r[n-1] the
// unique smallest sentinel) lexicographically. O(n log n).
vector<int> build_suffix_array(const vector<int>& r, int n, int m) {
    vector<int> sa(n), x(r), y(n, 0), ws(max(m, n), 0);

    // Initial order by one code.
    for (int i = 0; i < m; i++) ws[i] = 0;
    for (int i = 0; i < n; i++) ws[x[i]]++;
    for (int i = 1; i < m; i++) ws[i] += ws[i - 1];
    for (int i = n - 1; i >= 0; i--) sa[--ws[x[i]]] = i;

    for (long long j = 1, p = 1; p < n; j *= 2) {
        // Build the order by the later half, then stably sort by the earlier half.
        p = 0;
        for (long long i = max((long long)n - j, 0LL); i < n; i++) y[p++] = (int)i;
        for (int i = 0; i < n; i++)
            if (sa[i] >= j) y[p++] = sa[i] - (int)j;

        vector<int> wv(n);
        for (int i = 0; i < n; i++) wv[i] = x[y[i]];
        for (int i = 0; i < m; i++) ws[i] = 0;
        for (int i = 0; i < n; i++) ws[wv[i]]++;
        for (int i = 1; i < m; i++) ws[i] += ws[i - 1];
        for (int i = n - 1; i >= 0; i--) sa[--ws[wv[i]]] = y[i];

        // Reuse the arrays: y becomes the previous key array, x receives new keys.
        swap(x, y);
        p = 1;
        x[sa[0]] = 0;
        for (int i = 1; i < n; i++) {
            int a = sa[i - 1], b = sa[i];
            int left_a = (a + j < n) ? y[a + (int)j] : -1;
            int left_b = (b + j < n) ? y[b + (int)j] : -1;
            bool same = (y[a] == y[b]) && (left_a == left_b);
            if (same) x[b] = (int)p - 1;
            else x[b] = (int)p++;
        }
        m = (int)p;
    }
    return sa;
}

// height[i] = LCP(Suffix(sa[i-1]), Suffix(sa[i])); height[0] = 0. O(n).
vector<int> build_height(const vector<int>& s, const vector<int>& sa) {
    int n = (int)s.size();
    vector<int> rank(n, 0), height(n, 0);
    for (int i = 0; i < n; i++) rank[sa[i]] = i;  // inverse of sa

    int k = 0;
    for (int i = 0; i < n; i++) {
        int pos = rank[i];
        if (pos == 0) { k = 0; continue; }
        if (k) k--;
        int j = sa[pos - 1];
        while (i + k < n && j + k < n && s[i + k] == s[j + k]) k++;
        height[pos] = k;
    }
    return height;
}

int main() {
    string text;
    getline(cin, text);

    // Append a sentinel (chr 0) strictly smaller than every real character,
    // then rank-compress characters so keys are dense in [0, m).
    vector<int> s;
    s.reserve(text.size() + 1);
    for (unsigned char c : text) s.push_back((int)c + 1);  // shift so 0 is free
    s.push_back(0);                                        // sentinel = 0, smallest
    int n = (int)s.size();

    vector<int> sorted_codes = s;
    sort(sorted_codes.begin(), sorted_codes.end());
    sorted_codes.erase(unique(sorted_codes.begin(), sorted_codes.end()),
                       sorted_codes.end());
    int m = (int)sorted_codes.size();
    vector<int> r(n);
    for (int i = 0; i < n; i++)
        r[i] = (int)(lower_bound(sorted_codes.begin(), sorted_codes.end(), s[i]) -
                     sorted_codes.begin());

    vector<int> sa = build_suffix_array(r, n, m);
    vector<int> height = build_height(s, sa);

    cout << n << "\n";
    for (int i = 0; i < n; i++) cout << sa[i] << " \n"[i == n - 1];
    for (int i = 0; i < n; i++) cout << height[i] << " \n"[i == n - 1];
    return 0;
}
```

The whole chain: comparing suffixes as opaque strings costs $O(n)$ per comparison because they share long prefixes and the work is thrown away, so instead I rank every suffix by a bounded-length prefix and double that length, since a length-$2\ell$ prefix is the pair of length-$\ell$ ranks of the suffix and the one $\ell$ further on; each doubling sorts those pairs in linear time with a stable counting sort — and the second key needs no sort at all because the previous round's $SA$, read with a shift, already lists the suffixes in second-key order — so the sort is $O(n\log n)$, an appended smallest sentinel makes every suffix strictly ordered, missing second halves compare as one synthetic smallest key, and ranks are recomputed by adjacent-pair equality with an early stop when all ranks separate; then, indexing the adjacent-LCP array by start position turns "drop one leading character" into the inequality $h[i] \ge h[i-1]-1$, which lets a single position-order pass build the height array in $O(n)$.
