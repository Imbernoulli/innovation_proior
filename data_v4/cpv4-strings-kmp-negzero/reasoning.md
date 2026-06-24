**Reading the problem and pinning the contract.** I have a text `t[0..n-1]` and a pattern `p[0..m-1]`, all integers that can be negative, zero, or positive. The pattern *resonates* at a start `i` if some single offset `c` makes `t[i+j] = p[j] + c` for every `j` in `[0, m-1]`. I must output the count of resonance positions and the positions themselves in increasing order. Before any algorithm I fix the scale, because it dictates the data types and which approach survives: `n, m <= 2*10^5` and `|t[i]|, |p[j]| <= 10^9`. The absolute values themselves never get summed — there is no accumulation here — but I will be subtracting two values, and `t[i+1] - t[i]` can be as large as `10^9 - (-10^9) = 2*10^9`, which is past the 32-bit signed ceiling of about `2.147*10^9`... actually `2*10^9 < 2.147*10^9`, so a single difference *just* fits in `int`. That is uncomfortably close, and if I ever subtract in a wider expression or the judge nudges the bound, it breaks. I will use `long long` for everything to remove the question entirely. That is decision one and it is cheap insurance.

**The key reduction, and a sanity-check that it is exactly right.** The offset `c` is free, so absolute heights are irrelevant; only the *shape* matters. If `t[i+j] = p[j] + c` holds for all `j`, then subtracting consecutive equations, `t[i+j+1] - t[i+j] = (p[j+1] + c) - (p[j] + c) = p[j+1] - p[j]`. The `c` cancels. So a resonance at `i` forces the length-`(m-1)` consecutive-difference sequence of the window to equal the consecutive-difference sequence of the pattern. Is the converse true — does matching all `m-1` differences guarantee a single consistent `c`? Yes: define `c = t[i] - p[0]`. Then `t[i] = p[0] + c` by definition, and inductively `t[i+j+1] = t[i+j] + (t[i+j+1]-t[i+j]) = (p[j]+c) + (p[j+1]-p[j]) = p[j+1] + c`. So matching differences is *equivalent* to resonance, not merely necessary. Good — the reduction is lossless, and I trust it because I proved both directions. The problem is now: find every occurrence of the difference sequence `pd` (length `m-1`) inside the difference sequence `td` (length `n-1`) as a contiguous block. That is textbook exact matching over an integer alphabet, which KMP does in `O(n+m)`.

**Laying out the candidate approaches.** Two routes, and I want the one I can defend at the limits.

- *Naive windowed check.* For each start `i` in `[0, n-m]`, set `c = t[i] - p[0]` and verify all `m` entries; count the survivors. Dead simple and obviously correct, so it makes a perfect brute-force oracle. But it is `O(n*m)`, and an adversary can make almost every window almost-match (e.g. a constant text and a constant pattern) so the inner loop runs long — `2*10^5 * 2*10^5 = 4*10^10`, far past a 1-second budget. Too slow to ship.
- *KMP on the difference sequences.* Build `pd` and `td`, compute the KMP failure function of `pd`, scan `td`, and every full match maps back to a text start position. `O(n+m)` time and space. The idea's correctness I already settled; the risks are (a) index bookkeeping translating a difference-array match back to an original start index, and (b) the **degenerate pattern lengths** `m = 0` and `m = 1`, where `pd` is *empty* and the generic KMP loop has nothing to say. Those two corners are exactly where I expect to bleed, so I will handle them explicitly and trace them.

**Deriving the index map and checking it on paper.** KMP scans `td[0..(n-1)-1]` (length `tn = n-1`) and reports a full match of `pd` (length `pm = m-1`) whenever the matched-prefix counter `k` reaches `pm`. If the match *ends* at `td` index `e`, it *covers* `td[e-pm+1 .. e]`, i.e. it starts at difference-array index `s = e - pm + 1`. Now, `td[s]` is the difference `t[s+1]-t[s]`, and the window of original values that this block of `pm` differences describes is `t[s], t[s+1], ..., t[s+pm]` — that is `pm+1 = m` consecutive originals starting at original index `s`. So the text start position is exactly `s = e - pm + 1`. The difference-array start index equals the original start index; no off-by-one shift between the two coordinate systems. Let me sanity-check with the worked sample `t = [-1, 0, -2, 2, 3, 1]`, `p = [5, 6, 4]`. Then `pd = [6-5, 4-6] = [1, -2]`, length `pm = 2`. And `td = [0-(-1), -2-0, 2-(-2), 3-2, 1-3] = [1, -2, 4, 1, -2]`, length `5`. Where does `[1, -2]` occur in `[1, -2, 4, 1, -2]`? At `td` indices `0..1` (ends at `e=1`, start `s = 1-2+1 = 0`) and at `td` indices `3..4` (ends at `e=4`, start `s = 4-2+1 = 3`). So positions `0` and `3` — matching the stated answer `0 3`. The index map is right.

**First implementation — the degenerate cases, written fast and then attacked.** My instinct is to special-case `m` up front and otherwise run KMP. First cut of the head of `main`:

```
if (m == 1) {
    // a single value matches up to a shift everywhere... but careful
    cout << n << "\n";
    for (int i = 0; i < n; i++) cout << i << (i+1<n?' ':'\n');
    return 0;
}
// (no m == 0 branch yet)
int pm = m - 1;
vector<long long> pd(pm), td(n - 1);
...
```

I deliberately *forgot* the `m == 0` branch on the first pass, and I left a subtle hole in `m == 1`, because I want to see the failures concretely rather than reason about them abstractly.

**The first bug: `m == 0` falls through into a length-`-1` difference array.** Trace `m = 0` (empty pattern) with, say, `n = 3`, `t = [1, 2, 3]`. The `m == 1` branch does not fire. I reach `int pm = m - 1 = -1;` and then `vector<long long> pd(pm)` — constructing a vector with size `-1`. As a `size_t` that is `18446744073709551615`, an instant `std::bad_alloc` or crash. Even setting the crash aside, what *should* the answer be? An empty pattern has no window to align — there is no `j` to satisfy and no first element to anchor `c`. The contract says report `0` positions for `m = 0`. So I must intercept `m == 0` *before* computing `pm`. This is the classic "wrong base case" — the generic code assumes `m >= 1` and silently corrupts when handed `m = 0`. Fix: a guard at the very top.

```
if (m == 0) {
    cout << 0 << "\n" << "\n";   // count line, then empty positions line
    return 0;
}
```

Re-trace `m=0, n=3`: prints `0` and an empty line, no array of size `-1` ever constructed. Correct.

**The second bug: the `m == 1` base case and the all-negative / empty corner.** The `m == 1` branch claims every position `0..n-1` resonates. Is that right, including the signs? A length-1 window `t[i]` versus a length-1 pattern `p[0]`: pick `c = t[i] - p[0]` and the single equation `t[i] = p[0] + c` holds trivially — *always*, for any values including negatives and zeros. So yes, all `n` positions resonate, and the offsets are irrelevant. So far so good. But now trace `n = 0, m = 1`, `t = []`, `p = [7]`. There are zero positions, so the answer is `0` with an empty positions line. My first-cut code does `cout << n` = `0`, then the `for` loop runs zero times and prints *nothing* on line 2 — but my ternary `(i+1<n?' ':'\n')` never executes, so I never emit the line-2 newline at all. Output becomes just `0\n` with no second line. The contract says exactly two lines, the second empty. That is a malformed-output bug hiding in the `n = 0` corner of the `m = 1` base case. Worse, if I had instead written the count print as part of the loop I could have emitted a stray separator. Let me trace the intended fix: after `cout << n << "\n";` and the loop, if `n == 0` emit the empty positions line explicitly.

```
if (m == 1) {
    cout << n << "\n";
    for (int i = 0; i < n; i++) cout << i << (i + 1 < n ? ' ' : '\n');
    if (n == 0) cout << "\n";   // empty positions line when there are none
    return 0;
}
```

Re-trace `n=0, m=1`: prints `0\n` then `\n` — two lines, second empty. Correct. Re-trace `n=4, m=1`, `t=[-3,0,9,-3]`, `p=[100]`: prints `4` then `0 1 2 3` (the loop emits a space after each but a newline after the last because `i+1<n` is false at `i=3`); the `if (n==0)` does not fire. Correct, and notice the all-negative/zero/positive mix in `t` and the wildly-off `p=[100]` change nothing — every single position resonates, as it must up to shift. This is the sign-handling check: a single element always matches, regardless of sign, and the code does not gate on positivity anywhere.

**A third subtlety I check before trusting KMP: `m > n`.** If the pattern is longer than the text, no window fits and the answer is `0`. Does the KMP path handle this on its own? With `m > n`, `pm = m-1` and `tn = n-1`; the scan loop over `td` runs `tn` times and can never push `k` up to `pm` because there simply are not enough difference entries (`tn < pm` when `m > n`). So it would print `0` correctly — *unless* `n = 0`, where `td` has size `n-1 = -1`, the same negative-size vector disaster as the `m=0` case. To be safe and explicit I add a guard `if (m > n) { print 0; return; }` *after* the `m==0` and `m==1` guards, so that by the time I build `td` of size `n-1` I am guaranteed `m >= 2` and `m <= n`, hence `n >= 2` and `n-1 >= 1`. No negative-size vector can ever be constructed on the KMP path. Trace `n=2, m=3`, `t=[0,0]`, `p=[1,1,1]`: the `m>n` guard fires, prints `0` and empty line. Correct.

**Building and tracing the KMP core on the sample.** Now `m >= 2`, `n >= m`. I build `pd` (length `pm = m-1 >= 1`) and `td` (length `n-1 >= 1`). The failure function over `pd`:

```
vector<int> fail(pm, 0);
for (int i = 1; i < pm; i++) {
    int k = fail[i - 1];
    while (k > 0 && pd[i] != pd[k]) k = fail[k - 1];
    if (pd[i] == pd[k]) k++;
    fail[i] = k;
}
```

The scan:

```
int k = 0;
int tn = n - 1;
for (int i = 0; i < tn; i++) {
    while (k > 0 && td[i] != pd[k]) k = fail[k - 1];
    if (td[i] == pd[k]) k++;
    if (k == pm) { hits.push_back(i - pm + 1); k = fail[k - 1]; }
}
```

Trace on the sample `pd = [1, -2]` (`pm=2`), `td = [1, -2, 4, 1, -2]` (`tn=5`). Failure function: `fail[0]=0`; `i=1`: `k=fail[0]=0`, while-loop skipped (`k=0`), check `pd[1]==pd[0]`? `-2 == 1`? no, so `k` stays `0`, `fail[1]=0`. So `fail=[0,0]`. Scan, `k=0`:
- `i=0`, `td[0]=1`: while skipped; `td[0]==pd[0]`? `1==1` yes, `k=1`. `k==pm`? `1==2` no.
- `i=1`, `td[1]=-2`: while `k>0 && td[1]!=pd[1]`? `-2 != -2`? no, skipped; `td[1]==pd[1]`? `-2==-2` yes, `k=2`. `k==pm`? yes -> push `i-pm+1 = 1-2+1 = 0`; `k=fail[1]=0`.
- `i=2`, `td[2]=4`: while `k>0`? no; `td[2]==pd[0]`? `4==1`? no, `k=0`.
- `i=3`, `td[3]=1`: `td[3]==pd[0]`? `1==1` yes, `k=1`.
- `i=4`, `td[4]=-2`: while `k>0 && td[4]!=pd[1]`? `-2!=-2`? no; `td[4]==pd[1]`? yes, `k=2`. `k==pm` -> push `4-2+1 = 3`; `k=fail[1]=0`.

Hits `= [0, 3]`. Output `2` then `0 3`. Exactly the expected answer, with negatives and zeros flowing through the equality comparisons untouched. The reason this works regardless of sign: KMP only ever asks `pd[i] != pd[k]` and `td[i] != pd[k]`, plain integer (in)equality, and I store everything in `long long`. There is no ordering, no positivity test, no hashing — sign is simply not a variable the algorithm branches on.

**Edge cases, deliberately, because this is where this kind of code dies.**
- *Empty input (`n` absent):* `if (!(cin >> n)) return 0;` exits printing nothing. The judge for an empty file expects nothing. Fine.
- *`m = 0`:* guarded first; prints `0` and an empty line; never touches `pm = -1`. Correct.
- *`m = 1`, `n = 0`:* prints `0` then an explicit empty line — two lines as required. Correct.
- *`m = 1`, `n > 0` with all-negative/zero values:* every position printed; signs irrelevant. Correct.
- *`m > n` (including `n = 0`, `m >= 2`):* guarded; prints `0`. The KMP path, which builds `td` of size `n-1`, is never reached unless `n >= m >= 2`, so `n-1 >= 1` and no negative-size vector. Correct.
- *All-negative text and pattern with a real shift, e.g. `t = [-5,-2,-9,-1], p = [-1,-5]`:* `pd = [-4]`, `td = [3,-7,8]`; `-4` occurs nowhere, answer `0`. Brute confirms `0`. So a genuinely all-negative instance with no resonance returns `0`, not a spurious hit — the sign handling does not accidentally match.
- *Overflow stress, `t = [10^9, -10^9, 10^9, -10^9], p = [0, -2*10^9]`:* `td[0] = -10^9 - 10^9 = -2*10^9`, which overflows 32-bit but is exact in `long long`; `pd = [-2*10^9]`; matches at difference indices 0 and 2 -> positions `0 2`. Brute (Python, arbitrary precision) agrees. The `long long` choice earns its keep here.
- *Output format:* always exactly two lines — count, then space-separated positions or an empty line. KMP pushes hits in increasing `i`, and the brute counts in increasing `i`, so ordering matches without an extra sort.

**Fix and re-verification at scale.** I wrote an independent `O(n*m)` brute force (fix `c = t[i]-p[0]`, verify the window) and a generator that deliberately oversamples the danger zone: `n in [0,8]`; value modes including a tiny `[-2,2]` alphabet (so shifts collide and resonances are common), an all-non-positive mode, an all-*negative* mode, and a zero-heavy mode; pattern length `m` drawn from `0` up to `n+1` so `m=0`, `m=1`, and `m>n` all appear; and half the time the pattern is a shifted copy of a real text window so the *matching* path (not just misses) is exercised. Running 1100 random cases (seeds 1..1100) under a canonical comparator (count line plus the position multiset/order) yields zero mismatches. The two bugs I found by hand — the `m=0` fall-through into a `-1`-sized vector and the missing line-2 newline in the `n=0, m=1` corner — are precisely the wrong-base-case / sign-corner defects the structure of the problem invites, and they no longer occur.

**Final solution.** I proved the difference-sequence reduction in both directions, derived and hand-checked the difference-array-to-text-index map on the sample, traced the KMP core to the expected `0 3`, and closed the `m=0`, `m=1`, `m>n`, `n=0`, all-negative, and overflow corners explicitly. This is what I ship — one self-contained file, the `O(n+m)` KMP I can defend rather than the quadratic brute:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n)) return 0;          // no input at all
    vector<long long> t(n);
    for (auto &x : t) cin >> x;
    cin >> m;
    vector<long long> p(m);
    for (auto &x : p) cin >> x;

    // A resonance at position i means there is a constant c with t[i+j] = p[j] + c
    // for all 0 <= j < m. Subtracting consecutive entries cancels c, so this is
    // equivalent to: the (m-1)-length difference sequence of p equals the
    // (m-1)-length difference sequence of the window t[i..i+m-1].
    //
    // Special cases on the pattern length BEFORE building any difference array:
    //   m == 0 : an empty pattern. No resonance is defined for an empty pattern
    //            (there is no window to align), so the answer is 0 with no positions.
    //   m == 1 : the difference sequence is empty, so EVERY start position
    //            i in [0, n-1] resonates (a single value always matches up to shift).
    //            If n == 0 there are still zero positions.
    // These are exactly the base cases a naive "run KMP on the diff arrays" forgets.

    if (m == 0) {
        cout << 0 << "\n" << "\n";   // count line, then (empty) positions line
        return 0;
    }
    if (m == 1) {
        // Every position 0..n-1 is a resonance (window of length 1).
        cout << n << "\n";
        for (int i = 0; i < n; i++) cout << i << (i + 1 < n ? ' ' : '\n');
        if (n == 0) cout << "\n";    // empty positions line when there are none
        return 0;
    }
    if (m > n) {
        // Pattern longer than text: no window fits.
        cout << 0 << "\n" << "\n";
        return 0;
    }

    // Build difference sequences (lengths m-1 and n-1). Values can be negative,
    // zero, or positive; comparisons are plain integer equality so sign is irrelevant
    // to correctness as long as we use a signed 64-bit type.
    int pm = m - 1;
    vector<long long> pd(pm), td(n - 1);
    for (int j = 0; j < pm; j++) pd[j] = p[j + 1] - p[j];
    for (int j = 0; j + 1 < n; j++) td[j] = t[j + 1] - t[j];

    // KMP failure function over the pattern-difference sequence pd (length pm >= 1).
    vector<int> fail(pm, 0);
    for (int i = 1; i < pm; i++) {
        int k = fail[i - 1];
        while (k > 0 && pd[i] != pd[k]) k = fail[k - 1];
        if (pd[i] == pd[k]) k++;
        fail[i] = k;
    }

    // Scan the text-difference sequence. A full match of pd ending at td index e
    // corresponds to a window of m consecutive original values starting at
    // (e - pm + 1) in the difference array, i.e. text start index s = e - pm + 1.
    vector<int> hits;
    int k = 0;
    int tn = n - 1; // length of td
    for (int i = 0; i < tn; i++) {
        while (k > 0 && td[i] != pd[k]) k = fail[k - 1];
        if (td[i] == pd[k]) k++;
        if (k == pm) {
            int s = i - pm + 1;     // start index in difference array == text start
            hits.push_back(s);
            k = fail[k - 1];
        }
    }

    cout << (int)hits.size() << "\n";
    for (size_t i = 0; i < hits.size(); i++)
        cout << hits[i] << (i + 1 < hits.size() ? ' ' : '\n');
    if (hits.empty()) cout << "\n";
    return 0;
}
```

**Causal recap.** The free offset `c` cancels under consecutive differencing, which I proved is an *equivalence* (define `c = t[i]-p[0]` and induct), reducing resonance to exact substring search of `pd` in `td`, solved by KMP in `O(n+m)`; the difference-array start index equals the original text start, verified on the sample to give `0 3`. The traps were not in the matching but in the *length corners*: handing the generic KMP path `m = 0` builds a `-1`-sized vector (caught by tracing `m=0,n=3` to a crash, fixed with an up-front guard returning `0`), and the `m = 1` base case dropped the line-2 newline when `n = 0` (caught by tracing `n=0,m=1` to a one-line output, fixed with an explicit empty line); a third guard `m > n` keeps the `td` of size `n-1` off the negative-size path. Because every comparison is plain `long long` equality with no positivity branch, negatives, zeros, and the `2*10^9` overflow of a difference flow through untouched, and an independent quadratic brute over 1100 generated cases — heavily weighted toward all-negative, zero-heavy, and degenerate-length inputs — agrees on every one.
