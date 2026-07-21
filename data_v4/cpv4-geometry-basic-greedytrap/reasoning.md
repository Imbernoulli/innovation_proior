**The trap here is the anchoring rule, so I fix the types and then try to break the obvious greedy.** I have `n` houses at integer positions on a line, and a lamp at real `s` lights the closed window `[s, s+L]`; I want the fewest length-`L` windows covering every house. Two facts about the setup decide the solution before I write a loop. First the scale: `n <= 2*10^5`, `L <= 2*10^9`, `|x[i]| <= 10^9`. The lamp count is at most `n` and fits an `int`, but the arithmetic `x[i] + L` reaches `10^9 + 2*10^9 = 3*10^9`, past the signed 32-bit cap of `~2.1*10^9`. So `x`, `L`, and any `x[i]+L` must be `long long` — an `int` there is a silent wrong answer on the large tests, not a crash I would catch. Second, the lamp slides continuously while houses are integers, and the window is closed, so a house exactly on an endpoint is lit; that boundary will decide a comparison operator later.

**Two routes: a greedy to submit, a DP to check it with.** Sort the houses and sweep — repeatedly take the leftmost still-dark house, drop one lamp covering it, skip everything that lamp reaches. `O(n log n)`, a few lines. Everything rides on *where* the lamp sits relative to that dark house, which is exactly the trap this problem is built around. The alternative — a DP that, to cover the leftmost uncovered house, tries every integer left-anchor `s` in `[min(x)-1, max(x)+1]` and recurses — is obviously correct, because integer coordinates and integer `L` mean every distinct coverage pattern is realized by some integer anchor, but it is `O(n*range)` and hopeless at `n=2*10^5`. So the DP is my oracle, not my submission.

**Centering the lamp on the dark house looks fair and is wrong.** The symmetric-feeling choice puts the dark house at the window's center. On the sample `x=[2,3,9,9,14,20]`, `L=5`: center on `2` gives `[-0.5,4.5]`, lighting `2,3`; center on `9` gives `[6.5,11.5]`, lighting both `9`s but not `14`; then `14`; then `20` — **4** lamps. Left-anchor instead: `[2,7]` lights `2,3`; `[9,14]` lights `9,9,14`; `[20,25]` lights `20` — **3** lamps, matching the context's stated answer. The reason centering loses is concrete: it spends half the window, `L/2`, on empty road to the *left* of the leftmost dark house, where nothing needs lighting. That wasted half is reach that could have stretched right and swallowed `14` in the same lamp.

**The right rule, and why it is optimal.** To cover the leftmost dark house `p` while reaching as far right as possible, put the lamp's *left edge* at `p`: window `[p, p+L]`. Any lamp covering `p` has `s <= p`, so its right end `s+L <= p+L`; `s=p` maximizes that right end among all lamps still covering `p`. So `[p, p+L]` covers a superset of the right-side houses any other `p`-covering window could. Replacing an optimal solution's first lamp by `[p, p+L]` never increases the count; induct on the rest. That is the greedy: sort; while houses remain, take the leftmost dark `p`, place `[p, p+L]`, skip every house `<= p+L`, increment.

**The naive skip loop reads off the end of the array.** First cut of the core:

```
sort(x.begin(), x.end());
long long lamps = 0;
int i = 0;
while (i < n) {
    long long right = x[i] + L;
    lamps++;
    while (x[i] <= right) i++;   // skip everything this lamp covers
}
```

The inner `while (x[i] <= right) i++` advances `i` and then re-reads `x[i]` to test, with no `i < n` guard. On a single house `n=1`, `x=[5]`, `L=3`: outer `right=8`, `lamps=1`; inner reads `x[0]=5<=8`, `i=1`; then re-tests `x[1]` with `i=1=n` — out of bounds, undefined behavior. The fix is to put the bound test first and lean on `&&` short-circuit so `x[i]` is never evaluated at `i==n`: `while (i < n && x[i] <= right) i++;`. Both loops then stop cleanly on `[5]` at answer `1`.

**The closed boundary forces `<=`, not `<`.** A house exactly at `p+L` is lit and must be absorbed. On `x=[0,5]`, `L=5`, both lie inside `[0,5]` so the answer is `1`: `right=5`, `lamps=1`; the inner loop skips `0`, then `5<=5` sends `i=2` and stops — one lamp. With `x[i] < right` the house at `5` stays dark and costs a second lamp, a wrong `2`. So `<=` is load-bearing, matching the closed segment.

**`L=0` and duplicates fall out with no special case.** `L=0` makes each lamp light a single point, so the answer is the number of *distinct* positions. On `x=[4,4,7]`, `L=0`: lamp `[4,4]` skips both `4`s (`4<=4`), stops at `7`; lamp `[7,7]` gives `2`, the distinct-position count. Duplicates at one spot share a lamp, and the leftmost-dark anchoring plus `<=` handle both together.

**Remaining edges.** `n=0`: the sweep never runs, but I add an explicit early `0` so I neither sort an empty vector nor lean on the read-guard alone. Very large `L` (say `2*10^9`) makes the first lamp's `right=x[0]+L` cover everything for answer `1` — and this is exactly the `x[0]+L` up to `3*10^9` that forced `long long`. Negative coordinates behave identically; the centering trap reappears for them and the left-anchor rule still wins.

**Cross-check against the oracle.** I run the greedy against the exhaustive DP on a few hundred random small instances — mixed signs, tiny and huge `L`, heavy duplicates, `L=0` — and get zero mismatches, backing the exchange proof empirically. That is the left-anchor greedy: one `O(n log n)` sweep, `long long` arithmetic, and the `i < n && x[i] <= right` test doing double duty as bound guard and closed-boundary check; the full program is in the answer.
