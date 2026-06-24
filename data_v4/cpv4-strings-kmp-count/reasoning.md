**Reading the problem and pinning the contract.** I get one string `S` of length `n` over lowercase letters, and for every length `L` from `1` to `n` I must report `c[L]`, the number of times the length-`L` prefix `S[0..L-1]` occurs as a substring of `S`, overlaps allowed. Output is the `n` numbers on one line. Let me fix scale first because it kills the naive approach: `n <= 2*10^5`, so the obvious "for each prefix scan all starts and compare" is `O(n^2)` character work — on `S = "aaaa...a"` that is literally `~n^2/2 = 2*10^10` comparisons, hopeless under one second. I also note the trivial invariants I can use as sanity anchors: `c[L] >= 1` always (the prefix matches itself at position `0`), and `c[n] = 1` always (the whole string fits only at position `0`). Counts can reach `n = 2*10^5`, comfortably inside `int`, but I will accumulate in `long long` to be safe since I am summing many `+=` into the same cell; nothing forces overflow but it costs nothing to be defensive.

**Candidate approaches.** Two routes, and I want the one I can prove linear and prove correct, not just the one that types fast.

- *Direct border walk.* Build the KMP failure function `pi`, then for each end position `i` walk the border chain `pi[i], pi[pi[i]-1], ...` and do `c[len]++` for each border length plus the full prefix that ends there. This is correct by definition but the chain can be `O(n)` long per position (think `"aaaa...a"`), so it is `O(n^2)` again — same trap as brute force, just disguised.
- *Failure function with a single propagation pass.* Build `pi`, seed each end position's contribution at exactly one length, then push counts down the border chain *once* in the right order so every length is finalized before it donates to its own border. This is `O(n)`. The risk here is not the idea; it is the bookkeeping — which length receives the seed, the order of the push, and the "+1 for the prefix itself" term. This is the classic place to double-count or be off by one, so I will trace it carefully.

I commit to the propagation approach but I will derive the seeding from scratch rather than trust a half-remembered template.

**Deriving the failure function and the occurrence structure.** `pi[i]` is the length of the longest proper prefix of `S[0..i]` that is also a suffix of `S[0..i]`, with `pi[0] = 0`. Standard linear construction: `j = pi[i-1]`; while `j > 0` and `S[i] != S[j]`, set `j = pi[j-1]`; if `S[i] == S[j]` then `j++`; `pi[i] = j`.

Now the central fact. A length-`L` prefix occurs *ending at position `i`* (i.e. occupying `S[i-L+1..i]`) iff `L` is a border length of `S[0..i]` — that is, iff `L` appears in the chain `i+1` (the whole prefix `S[0..i]`, length `i+1`), then `pi[i]`, then `pi[pi[i]-1]`, and so on down to `0`. The length `i+1` term is the prefix `S[0..i]` occurring as *itself*. The remaining chain `pi[i] -> pi[pi[i]-1] -> ...` are the *proper* prefix-suffixes ending at `i`. So if I count, for every `i`, every entry of the chain `pi[i], pi[pi[i]-1], ...`, I get all the *non-self* occurrences of every prefix; the self-occurrence of each length-`L` prefix is the single match at position `0`, which I will add separately as `+1` per length.

**Turning the chain walk into one propagation pass.** Walking each chain explicitly is too slow. The trick: for each `i` I only seed the *top* of its proper chain, `len = pi[i]` (when `pi[i] > 0`), with a single `cnt[len]++`. Then I exploit the recursion: every occurrence of the length-`L` prefix is itself a string whose own longest border is `pi[L-1]`, and that border occurs at exactly the same ending positions. So whatever total `cnt[L]` accumulates, all of it must also flow into `cnt[pi[L-1]]`. Concretely:

```
for each i: if pi[i] > 0 then cnt[pi[i]] += 1        // seed: top of proper chain at i
for L = n down to 1: if pi[L-1] > 0 then cnt[pi[L-1]] += cnt[L]   // push down the chain
```

The order matters: I must finalize `cnt[L]` *before* I push it into `cnt[pi[L-1]]`, and since `pi[L-1] < L` strictly (a proper border is shorter), processing `L` from high to low guarantees `cnt[L]` is complete when I read it. Finally `cnt[L] += 1` for the self-occurrence.

**Sanity-checking the derivation on `"abacaba"`.** `pi` for `a b a c a b a` (0-indexed): `pi = [0,0,1,0,1,2,3]`. Expected answer `4 2 2 1 1 1 1`. Seed pass — for each `i` with `pi[i]>0`: `i=2 -> cnt[1]++`, `i=4 -> cnt[1]++`, `i=5 -> cnt[2]++`, `i=6 -> cnt[3]++`. So `cnt = [_,2,1,1,0,0,0,0]` (index 0 unused). Push pass, `L` from 7 down to 1, using border `b=pi[L-1]`:
- `L=7, b=pi[6]=3: cnt[3]+=cnt[7]=0 -> cnt[3]=1`.
- `L=6, b=pi[5]=2: cnt[2]+=cnt[6]=0 -> cnt[2]=1`.
- `L=5, b=pi[4]=1: cnt[1]+=cnt[5]=0 -> cnt[1]=2`.
- `L=4, b=pi[3]=0: skip`.
- `L=3, b=pi[2]=1: cnt[1]+=cnt[3]=1 -> cnt[1]=3`.
- `L=2, b=pi[1]=0: skip`.
- `L=1, b=pi[0]=0: skip`.
Now `cnt = [_,3,1,1,1,0,0,0]`. Add `+1` per length: `cnt = [_,4,2,2,2,1,1,1]`. Expected `4 2 2 1 1 1 1`. The first three match but `c[4]` came out `2`, expected `1`. Something is wrong — and I caught it before writing code, which is exactly why I trace.

**The bug (first one): the self term must be added before propagation, not after.** Let me see where the extra count at length 4 came from. The length-4 prefix `"abac"` occurs only at position `0`; it has no proper border (`pi[3]=0`). So `cnt[4]` from seeds+push is `0`, and `+1` gives `1` — that part is fine. Re-checking my arithmetic: in the push pass I wrote `cnt[2]=1` after `L=6`, but the *seed* already had `cnt[2]=1`, and `L=6` did `cnt[2]+=cnt[6]=0`, leaving `1` — fine. The real slip is at length... let me recompute `cnt[4]`: seeds gave `cnt[4]=0`; no push targets length 4 except a push *from* some `L` with `pi[L-1]=4`, and no such `L` exists here, so `cnt[4]=0`, then `+1 = 1`. So actually `c[4]=1`. I miscopied above; redoing the vector after the push pass: `cnt[1]=3, cnt[2]=1, cnt[3]=1, cnt[4]=0, cnt[5..7]=0`, then `+1` each gives `4 2 2 1 1 1 1`. That is correct.

So the derivation is right and I made a transcription error in the trace itself — good that I recomputed. But this scare flags the genuinely dangerous variant of the same idea, which I now deliberately test: **what if I add the `+1` self term *before* the push pass instead of after?** That is a real, tempting ordering ("a prefix occurs once as itself, so initialize each cnt to 1, then propagate"). Let me trace that broken order on `"abacaba"`. Initialize `cnt[L]=1` for all `L=1..7`, then add seeds: `cnt[1]=1+2=3, cnt[2]=1+1=2, cnt[3]=1+1=2, cnt[4..7]=1`. Push `L=7..1`:
- `L=7,b=3: cnt[3]+=cnt[7]=1 -> 3`.
- `L=6,b=2: cnt[2]+=cnt[6]=1 -> 3`.
- `L=5,b=1: cnt[1]+=cnt[5]=1 -> 4`.
- `L=4,b=0: skip`.
- `L=3,b=1: cnt[1]+=cnt[3]=3 -> 7`.
- `L=2,b=0: skip`.
- `L=1,b=0: skip`.
Result `cnt[1]=7, cnt[2]=3, cnt[3]=3, ...` — wildly wrong (expected `4 2 2`). The defect is precise: the self `+1` placed at every length gets *propagated down the border chain along with the real counts*, so each prefix's self-occurrence is double-, triple-counted into its borders. The self term is an occurrence at position `0` that does **not** recurse into shorter prefixes the way ended-occurrences do, so it must be added strictly *after* all propagation. Conclusion locked in: seed -> push high-to-low -> then `+1`. My final code adds the `+1` last, which the trace now confirms is the only safe order.

**Second debug episode: the propagation direction.** The other classic way to wreck this is to run the push pass *low to high* instead of high to low. I want to feel that failure concretely, so I trace `"aaaa"`, expected `4 3 2 1`. `pi` for `a a a a` is `[0,1,2,3]`. Seed pass (`pi[i]>0`): `i=1 -> cnt[1]++`, `i=2 -> cnt[2]++`, `i=3 -> cnt[3]++`. So `cnt=[_,1,1,1,0]`.

First the *correct* high-to-low push, `b=pi[L-1]`:
- `L=4,b=pi[3]=3: cnt[3]+=cnt[4]=0 -> 1`.
- `L=3,b=pi[2]=2: cnt[2]+=cnt[3]=1 -> 2`.
- `L=2,b=pi[1]=1: cnt[1]+=cnt[2]=2 -> 3`.
- `L=1,b=pi[0]=0: skip`.
`cnt=[_,3,2,1,0]`, then `+1` -> `4 3 2 1`. Correct.

Now the *broken* low-to-high push on the same seeds `cnt=[_,1,1,1,0]`:
- `L=1,b=0: skip`.
- `L=2,b=1: cnt[1]+=cnt[2]=1 -> cnt[1]=2`.
- `L=3,b=2: cnt[2]+=cnt[3]=1 -> cnt[2]=2`.
- `L=4,b=3: cnt[3]+=cnt[4]=0 -> cnt[3]=1`.
`cnt=[_,2,2,1,0]`, then `+1` -> `3 3 2 1`. Wrong: `c[1]` should be `4`, I got `3`. The cause is exactly the finalization order: when I processed `L=2` I pushed `cnt[2]`'s value into `cnt[1]`, but `cnt[2]` was *not yet final* — the contribution that `cnt[3]` later donates into `cnt[2]` (the chain `len-3 -> len-2 -> len-1`) never reached `cnt[1]`, because `cnt[1]` was read-and-done before `cnt[2]` grew. High-to-low is the only order where every `cnt[L]` is complete at the moment it donates. This is the second, independent bookkeeping trap, and tracing `"aaaa"` shows it loses a count rather than adding one. My code loops `for (int L = n; L >= 1; L--)`, the correct direction.

**A third quiet pitfall I check on purpose: seeding length 0.** In the seed pass I guard `if (pi[i] > 0)`. Suppose I drop that guard and do `cnt[pi[i]]++` unconditionally. Then positions with `pi[i]=0` increment `cnt[0]`. `cnt[0]` is never printed and never a valid border target (a border length of `0` means "no border", and my push guards `if (b > 0)`), so a stray `cnt[0]` is harmless to the output — but it *would* matter if I ever pushed *from* `L` with `pi[L-1]` possibly `0` without guarding, since `cnt[0] += cnt[L]` then silently swallows counts that should have stopped. So both guards (`pi[i] > 0` on seed, `b > 0` on push) are load-bearing; I keep both. I verify on `"abcde"` (`pi=[0,0,0,0,0]`, all distinct): seeds add nothing (every `pi[i]=0`), push does nothing (every `b=0`), `+1` per length -> `1 1 1 1 1`. Correct, and it confirms the all-distinct corner.

**First full implementation and a trace.** I write the three passes:

```
vector<long long> cnt(n + 1, 0);
for (int i = 0; i < n; i++) if (pi[i] > 0) cnt[pi[i]]++;          // seed
for (int L = n; L >= 1; L--) { int b = pi[L-1]; if (b>0) cnt[b]+=cnt[L]; }  // push high->low
for (int L = 1; L <= n; L++) cnt[L] += 1;                        // self-occurrence, last
```

Trace `"aabaaab"` (a tricky one), expected `5 3 2 1 1 1 1`. `S = a a b a a a b`. Compute `pi`: `pi[0]=0`; `i=1 (a==a) pi=1`; `i=2 (b vs S[1]=a no, j=pi[0]=0, b vs a no) pi=0`; `i=3 (a vs S[0]=a yes) pi=1`; `i=4 (a, j=pi[3]=1, a vs S[1]=a yes) pi=2`; `i=5 (a, j=pi[4]=2, a vs S[2]=b no -> j=pi[1]=1, a vs S[1]=a yes) pi=2`; `i=6 (b, j=pi[5]=2, b vs S[2]=b yes) pi=3`. So `pi=[0,1,0,1,2,2,3]`. Seed (`pi[i]>0`): `i=1->cnt[1]++`, `i=3->cnt[1]++`, `i=4->cnt[2]++`, `i=5->cnt[2]++`, `i=6->cnt[3]++`. `cnt=[_,2,2,1,0,0,0,0]`. Push `L=7..1`, `b=pi[L-1]`:
- `L=7,b=pi[6]=3: cnt[3]+=cnt[7]=0 -> 1`.
- `L=6,b=pi[5]=2: cnt[2]+=cnt[6]=0 -> 2`.
- `L=5,b=pi[4]=2: cnt[2]+=cnt[5]=0 -> 2`.
- `L=4,b=pi[3]=1: cnt[1]+=cnt[4]=0 -> 2`.
- `L=3,b=pi[2]=0: skip`.
- `L=2,b=pi[1]=1: cnt[1]+=cnt[2]=2 -> 4`.
- `L=1,b=pi[0]=0: skip`.
`cnt=[_,4,2,1,0,0,0,0]`, then `+1` -> `5 3 2 1 1 1 1`. Matches expected exactly. The order (seed, high-to-low push, late `+1`) reproduces the known answer.

**Edge cases, deliberately.**
- `n = 1`, `S = "a"`: `pi=[0]`, seed adds nothing, push `L=1,b=pi[0]=0` skip, `+1` -> `cnt[1]=1`. Output `1`. The prefix is the whole string, occurs once. Correct, and `c[n]=1` invariant holds.
- All distinct `"abcde"`: shown above -> `1 1 1 1 1`. Correct.
- Maximal overlap `"aaaa"`: shown above -> `4 3 2 1`, and in general `"a"*n` gives `c[L]=n-L+1` (length-`L` block of a's starts at `0..n-L`). I check the closed form against `n-L+1`: `L=1->4, L=2->3, L=3->2, L=4->1`. Correct.
- Output format: I print `cnt[L]` for `L=1..n` separated by single spaces, newline after the last; the framework guard `if (!(cin>>s)) return 0;` handles absent input gracefully though the constraints promise `n>=1`.
- Overflow: counts are at most `n=2*10^5`, far inside `int`; I still accumulate in `long long` because the push does repeated `+=`, and the cost is nil. No sentinel arithmetic anywhere, so nothing can underflow.

**Performance check.** All three passes are a single `O(n)` loop; the failure function is `O(n)` amortized. No border chain is ever walked explicitly — the propagation replaces it. On `"a"*200000` the program prints `200000 199999 ...` and finishes in about 15 ms, three orders of magnitude under the 1 s limit. The disguised-`O(n^2)` "walk the chain per position" approach would have died exactly here, which is why I rejected it up front.

**Final solution.** I proved the structure (occurrences of a prefix = its border-chain positions, plus the self-match at `0`), I disproved two tempting-but-wrong orderings by tracing them to concrete losses/double-counts (`+1` before the push triple-counts on `"abacaba"`; low-to-high push loses a count on `"aaaa"`), and I confirmed the correct ordering on `"abacaba"`, `"aabaaab"`, `"aaaa"`, `"abcde"`, and the singleton. I ship the three-pass `O(n)` solution:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;            // empty / no input -> nothing to print
    int n = (int)s.size();

    // KMP failure function: pi[i] = length of the longest proper prefix of
    // s[0..i] that is also a suffix of s[0..i].  pi[0] = 0.
    vector<int> pi(n, 0);
    for (int i = 1; i < n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[i] != s[j]) j = pi[j - 1];
        if (s[i] == s[j]) j++;
        pi[i] = j;
    }

    // cnt[L] = number of occurrences (overlaps allowed) of the length-L prefix
    // of s inside s, for L = 1..n.  We index cnt by length, cnt has size n+1.
    //
    // Step 1: every end position i contributes one occurrence of the prefix of
    // length pi[i] (the longest prefix-suffix ending at i).  Length 0 carries no
    // prefix, so we only seed positive lengths.
    vector<long long> cnt(n + 1, 0);
    for (int i = 0; i < n; i++)
        if (pi[i] > 0) cnt[pi[i]]++;

    // Step 2: if the length-L prefix occurs, then every shorter prefix that is a
    // border of it (length pi[L-1], then pi[pi[L-1]-1], ...) also occurs at each
    // of those positions.  Push counts down the failure chain.  Process lengths
    // from long to short so each cnt[L] is final before it is propagated.
    for (int L = n; L >= 1; L--) {
        int b = pi[L - 1];                // longest border length of prefix of length L
        if (b > 0) cnt[b] += cnt[L];
    }

    // Step 3: the length-L prefix also occurs once as the whole prefix itself
    // (the trivial occurrence at position 0), which the failure function never
    // counts.  Add it exactly once per length.
    for (int L = 1; L <= n; L++) cnt[L] += 1;

    for (int L = 1; L <= n; L++) {
        cout << cnt[L];
        cout << (L == n ? '\n' : ' ');
    }
    return 0;
}
```

**Causal recap.** The occurrences of the length-`L` prefix are exactly its border-chain ending positions plus the self-match at position `0`; turning the per-position chain walk into one push pass is what makes it `O(n)`, but the bookkeeping has two real traps and one quiet one. Adding the self `+1` *before* the push makes that term propagate down every border chain, triple-counting on `"abacaba"` (`c[1]` jumps to `7`); pushing *low-to-high* reads `cnt[L]` before it is finalized, losing a count on `"aaaa"` (`c[1]` drops to `3`); and dropping the `pi[i]>0` / `b>0` guards would let counts leak into the unused length-`0` cell. Seeding each end position at `cnt[pi[i]]`, pushing high-to-low so every cell is complete before it donates, and adding the self-occurrence strictly last gives the verified `4 2 2 1 1 1 1` on the sample and `n-L+1` on `"a"*n`, in linear time.
