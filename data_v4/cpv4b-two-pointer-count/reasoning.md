**Reading the problem and pinning the contract.** I have `n` tuning forks with integer frequencies `f[0..n-1]`, and a tolerance band `[L, R]`. Two distinct forks are compatible when `L <= |f[i] - f[j]| <= R`, and I must count the **unordered** pairs `{i, j}` — each pair once — then report that count modulo `1 000 000 007`. Before any algorithm I fix the numbers, because they decide both the data types and the very need for the modulus. `n` reaches `10^6`, so the number of pairs can approach `C(10^6, 2) = ~5*10^11`. That fits in a 64-bit `long long` (which tops out near `9.2*10^18`), but the *output* is mandated modulo a prime, so the count is reduced at the end. Frequencies satisfy `|f[i]| <= 10^9` and the band bounds reach `2*10^9`; a difference like `f[j] - R` can be as low as `-10^9 - 2*10^9 = -3*10^9`, which overflows a 32-bit `int`. So every frequency, every band bound, and every window endpoint must be 64-bit. I will store `f`, `L`, `R`, and all derived bounds as `long long`. That decision is non-negotiable: an `int` here is a silent wrong-answer on the large or wide-band tests.

**Laying out the candidate approaches.** Two routes, and I commit to the one I can both make fast *and* count exactly.

- *Quadratic enumeration.* Loop over all `i < j`, test `L <= |f[i]-f[j]| <= R`, tally. It is transparently correct — it is exactly my mental definition — and it is my oracle. But at `n = 10^6` it is `~5*10^11` comparisons, far past a 2-second budget. Useless as the shipped solution, invaluable as the brute force I check against.
- *Sort plus two pointers.* Sorting the frequencies does not change which pairs are compatible — `|f[i]-f[j]|` is symmetric and order-independent — so I may sort freely. After sorting, for a fixed element treated as the *larger* one, the admissible partners form a contiguous block of the sorted array, and as the larger element moves right the block's two ends move right too. Two indices chase those ends. Cost is `O(n log n)`, dominated by the sort. The correctness questions are: the exact window endpoints, the `<` versus `<=` boundaries, and — the one that actually bites — counting each unordered pair once and not twice.

**Deriving the window after sorting.** Sort so that `f[0] <= f[1] <= ... <= f[n-1]`. Take any pair of sorted positions `i < j`; then `f[i] <= f[j]`, so `|f[i]-f[j]| = f[j] - f[i]`, and the pair is compatible exactly when `L <= f[j] - f[i] <= R`. I rearrange this into a constraint purely on `f[i]`:

- `f[j] - f[i] >= L`  becomes  `f[i] <= f[j] - L`;
- `f[j] - f[i] <= R`  becomes  `f[i] >= f[j] - R`.

So, fixing the larger position `j`, its compatible partners are the positions `i < j` whose value lies in the closed interval `[f[j] - R, f[j] - L]`. Because the array is sorted, "values in `[f[j]-R, f[j]-L]`" is a contiguous run, and restricting to `i < j` just clips that run to the prefix before `j`. If I sum, over every `j`, the count of such partners in the prefix, I get every unordered pair exactly once — because each unordered pair `{i, j}` is counted only at its *larger* sorted endpoint. That last sentence is the whole anti-double-count argument; I will come back and verify I actually implemented it.

**Why two pointers and not a binary search each step.** For each `j` I need two prefix counts: how many prefix elements are `<= f[j] - L` (call the boundary `hi`), and how many are `< f[j] - R` (call it `lo`); the partners number `hi - lo`. As `j` increases, `f[j]` is nondecreasing (sorted), so both `f[j] - L` and `f[j] - R` are nondecreasing, hence both `hi` and `lo` only ever advance. Two monotone pointers sweep the array once in `O(n)` after the `O(n log n)` sort. I could binary-search each `j` for `O(n log n)` total too, but the two-pointer sweep is simpler to reason about and I want to *see* the monotonicity to trust the count.

**Numeric self-check of the per-`j` decomposition.** Before writing code I verify the claim "sum over `j` of (prefix partners of `j`) equals the true unordered count" on the documented sample, by hand. Sorted `f = [1, 4, 5, 8, 10, 13]`, `L = 2`, `R = 5`. For each `j` the window on values is `[f[j]-R, f[j]-L]`, intersected with the prefix `f[0..j-1]`:

- `j=0`, `f=1`: window `[-4, -1]`, prefix empty -> `0`.
- `j=1`, `f=4`: window `[-1, 2]`, prefix `{1}` -> `{1}` qualifies -> `1`.
- `j=2`, `f=5`: window `[0, 3]`, prefix `{1,4}` -> `{1}` -> `1`.
- `j=3`, `f=8`: window `[3, 6]`, prefix `{1,4,5}` -> `{4,5}` -> `2`.
- `j=4`, `f=10`: window `[5, 8]`, prefix `{1,4,5,8}` -> `{5,8}` -> `2`.
- `j=5`, `f=13`: window `[8, 11]`, prefix `{1,4,5,8,10}` -> `{8,10}` -> `2`.

Sum `= 0+1+1+2+2+2 = 8`. Independently, the compatible pairs by hand are `(1,4),(1,5),(4,8),(5,8),(5,10),(8,10),(8,13),(10,13)` — eight of them. The decomposition matches the documented answer `8`, so the per-`j`-at-the-larger-end accounting is sound on a concrete case, not just asserted.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first instinct is to make every fork the *center* of its band and count partners on both sides — the symmetric picture feels natural. I write this first cut:

```
sort(f.begin(), f.end());
long long total = 0;
for (int j = 0; j < n; j++) {
    // partners on the LOWER side: value in [f[j]-R, f[j]-L]
    int a = lower_bound(f, f[j]-R) ... ;   // first index with f >= f[j]-R
    int b = upper_bound(f, f[j]-L) ... ;   // first index with f >  f[j]-L
    // partners on the UPPER side: value in [f[j]+L, f[j]+R]
    int c = lower_bound(f, f[j]+L) ... ;
    int d = upper_bound(f, f[j]+R) ... ;
    total += (b - a) + (d - c);            // both sides, over the WHOLE array
}
```

The idea: for each fork, count every other fork whose gap is in band, lower-frequency and higher-frequency alike, and sum. Let me trace it on the sorted sample `[1,4,5,8,10,13]`, `L=2`, `R=5`. Lower-side counts (value in `[f[j]-R, f[j]-L]`, over the whole array) are `0,1,1,2,2,2` as I tabulated above, summing to `8`. Upper-side counts (value in `[f[j]+L, f[j]+R]`) are the mirror image: `f=1` -> `[3,6]` -> `{4,5}` -> `2`; `f=4` -> `[6,9]` -> `{8}` -> `1`; `f=5` -> `[7,10]` -> `{8,10}` -> `2`; `f=8` -> `[10,13]` -> `{10,13}` -> `2`; `f=10` -> `[12,15]` -> `{13}` -> `1`; `f=13` -> `[15,18]` -> `{}` -> `0`; summing to `8`. So `total = 8 + 8 = 16`.

**The bug.** The code returns `16`; the true answer is `8`. I have counted every unordered pair **twice** — once when I stood on its lower endpoint and looked up, and once when I stood on its higher endpoint and looked down. The pair `{1, 4}` (gap `3`) is counted at `j` for `f=1` (upper side, `4` is a partner) and again at `j` for `f=4` (lower side, `1` is a partner). The symmetric "count both sides" picture is exactly the trap: `|f[i]-f[j]|` is symmetric, so summing over both endpoints of every pair always yields `2 *` the unordered count. The naive fix would be to divide by two — but that is fragile under the modulus (`2` has an inverse mod the prime, fine here, yet it invites the habit of reducing before halving, after which the parity is gone and the halving is wrong), and worse, it papers over the cleaner structural fix. The right move is to count each pair at *one* canonical endpoint only. Since the array is sorted, I make the canonical endpoint the *larger* element `j` and look only at the **prefix** `i < j`. That drops the upper-side term entirely and clips the lower-side window to indices below `j`. Every pair is then seen exactly once, at its larger end. No division, no parity hazard.

**Second implementation and a second trace — getting the pointer bounds right.** I rewrite with two monotone pointers over the prefix. `hi` will be the count of prefix elements `<= f[j]-L`, and `lo` the count of prefix elements `< f[j]-R`; partners `= hi - lo`. Crucially both pointers are clamped to `< j` so they only ever see the prefix:

```
long long total = 0;
int lo = 0, hi = 0;
for (int j = 0; j < n; j++) {
    while (hi < j && f[hi] <= f[j] - L) hi++;
    while (lo < j && f[lo] <  f[j] - R) lo++;
    total += (long long)(hi - lo);
}
```

I trace the sorted sample `[1,4,5,8,10,13]`, `L=2`, `R=5`, watching `lo,hi`:

- `j=0` (`f=1`): `hi`: `hi<0` false -> stays `0`. `lo`: stays `0`. add `0-0=0`. total `0`.
- `j=1` (`f=4`): need prefix `<= 4-2=2`: `f[0]=1<=2` -> `hi=1`; `hi<1` now false. need `< 4-5=-1`: `f[0]=1<-1`? no -> `lo=0`. add `1-0=1`. total `1`.
- `j=2` (`f=5`): `<= 5-2=3`: `f[1]=4<=3`? no -> `hi=1`. `< 5-5=0`: `f[0]=1<0`? no -> `lo=0`. add `1`. total `2`.
- `j=3` (`f=8`): `<= 8-2=6`: `f[1]=4<=6` -> `hi=2`; `f[2]=5<=6` -> `hi=3`; `hi<3` false. `< 8-5=3`: `f[0]=1<3` -> `lo=1`; `f[1]=4<3`? no -> `lo=1`. add `3-1=2`. total `4`.
- `j=4` (`f=10`): `<= 10-2=8`: `f[3]=8<=8` -> `hi=4`; `hi<4` false. `< 10-5=5`: `f[1]=4<5` -> `lo=2`; `f[2]=5<5`? no -> `lo=2`. add `4-2=2`. total `6`.
- `j=5` (`f=13`): `<= 13-2=11`: `f[4]=10<=11` -> `hi=5`; `hi<5` false. `< 13-5=8`: `f[2]=5<8` -> `lo=3`; `f[3]=8<8`? no -> `lo=3`. add `5-3=2`. total `8`.

Final total `8` — matches. The bug case from the first cut (which gave `16`) now gives `8`, and it gives the right number for the reason I fixed: counting at the larger end over the prefix, never both sides.

**Probing the `<` versus `<=` boundary deliberately, because off-by-one in the endpoints is the other classic double/under count.** The window is the *closed* interval `[f[j]-R, f[j]-L]`. The count of prefix elements with value `<= f[j]-L` is `hi`; the count with value `< f[j]-R` is `lo`; their difference is the count with value in `[f[j]-R, f[j]-L]`. The asymmetry (`<=` for the upper boundary `hi`, `<` for the lower boundary `lo`) is intentional: the partner at exactly `f[j]-R` (gap exactly `R`) must be *included*, and the partner at exactly `f[j]-L` (gap exactly `L`) must be *included* — both endpoints are valid because the band is inclusive. `hi` with `<=` includes the value `f[j]-L`; `lo` with `<` excludes everything strictly below `f[j]-R` and so *keeps* the value `f[j]-R` in the `[lo, hi)` block. Let me confirm with a microcase where both bounds sit exactly on data: `f = [0, 2, 5]`, `L = 2`, `R = 5`. Pairs: `(0,2)` gap `2` in `[2,5]` yes; `(0,5)` gap `5` in `[2,5]` yes; `(2,5)` gap `3` yes -> answer `3`. Trace: `j=0` add 0. `j=1` (`f=2`): `<= 0`: `f[0]=0<=0` -> `hi=1`; `< -3`: `f[0]=0<-3`? no -> `lo=0`; add `1`. `j=2` (`f=5`): `<= 3`: `f[0]=0<=3`->`hi`... wait `hi` is already `1`, continue: `f[1]=2<=3` -> `hi=2`; `f[2]` not in prefix. `< 0`: `f[0]=0<0`? no -> `lo=0`; add `2-0=2`. total `0+1+2=3`. Correct — both the gap-exactly-`L` pair `(0,2)` and the gap-exactly-`R` pair `(0,5)` are included, so the inclusive endpoints are handled right.

**Edge cases, deliberately, because this is where counting code dies.**
- `n = 0`: the `for` loop never runs, `total = 0`, output `0 % MOD = 0`. The empty bench — correct. (Reading also degrades gracefully: `cin >> n` succeeds with `0`, then `f` is empty.)
- `n = 1`: a single fork has no partner; the loop runs once with `j=0`, both inner `while`s have `hi<0`/`lo<0` false, adds `0`. Output `0`. Correct.
- `L = 0`: equal-frequency forks are compatible (gap `0` is in band). With `L=0`, the upper boundary is `<= f[j]-0 = f[j]`, so prefix elements *equal* to `f[j]` are counted — exactly the equal-frequency partners. Tested on `f=[5,5,5,2]`, `L=R=0`: only the three `5`s pair among themselves, `C(3,2)=3`; the solution prints `3`. Correct.
- `R` never binds (e.g. `R = 2*10^9` while max gap is small): `lo` boundary `f[j]-R` is hugely negative, so `lo` stays `0`, and every prefix element with value `<= f[j]-L` counts — the lower bound alone governs. No overflow because `f[j]-R` is computed in `long long`.
- `L` never binds (`L = 0`) combined with large `R`: counts all unordered pairs `C(n,2)`, the densest case. On `n = 200000` equal values it returns `C(200000,2) mod p`, matching `n*(n-1)/2 mod p` exactly — verified.
- Overflow: `total` is `long long` and never exceeds `C(10^6,2) ~ 5*10^11 < 9.2*10^18`, so the running count never overflows; I reduce modulo the prime only once, at output. Window bounds `f[j]-R` reach `-3*10^9`, which is why they live in `long long`, not `int`.
- Modulus discipline: I keep `total` exact throughout and apply `% MOD` a single time at the end. There is no subtraction or division under the modulus, so no negative-residue or modular-inverse hazard — the double-count fix that removed the `/2` also removed the only place a modular division would have lurked.

**Final solution.** I convinced myself the *idea* is right by deriving the per-`j` window, checking the decomposition `0+1+1+2+2+2 = 8` against the eight hand-listed pairs, and — most importantly — by tracing the symmetric first cut to the precise `16 = 2*8` double-count and replacing it with prefix-only counting at the larger endpoint. I convinced myself the *code* is right by tracing the pointer sweep to `8`, by the `[0,2,5]` boundary microcase confirming inclusive endpoints, and by the `n=0/1`, `L=0`, never-binds, and `200000`-duplicate corners. That is what I ship — one self-contained file, the `O(n log n)` sort-and-sweep I can defend, with the modulus applied exactly once:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    long long L, R;
    cin >> L >> R;
    vector<long long> f(n);
    for (auto &x : f) cin >> x;

    const long long MOD = 1000000007LL;

    sort(f.begin(), f.end());

    // For an unordered pair {i, j} with sorted values f[i] <= f[j], the gap is
    // f[j] - f[i], and it is "compatible" iff L <= f[j] - f[i] <= R.
    // Fix the LARGER element j (right end). Its valid partners are the indices
    // i < j with  f[j] - R <= f[i] <= f[j] - L.
    // As j increases, both window bounds (f[j]-R and f[j]-L) are nondecreasing,
    // so two pointers lo, hi sweep the prefix [0, j) once. Counting only
    // partners to the LEFT of j makes each unordered pair counted exactly once.
    long long total = 0;       // exact count, fits in long long (<= n*(n-1)/2)
    int lo = 0, hi = 0;        // lo: first i with f[i] >= f[j]-R; hi: first i with f[i] > f[j]-L
    for (int j = 0; j < n; j++) {
        // hi: number of elements among the prefix with value <= f[j]-L.
        while (hi < j && f[hi] <= f[j] - L) hi++;
        // lo: number of elements among the prefix with value <  f[j]-R.
        while (lo < j && f[lo] <  f[j] - R) lo++;
        // valid partners i in [lo, hi): values in [f[j]-R, f[j]-L].
        total += (long long)(hi - lo);
    }

    cout << (total % MOD) << "\n";
    return 0;
}
```

**Causal recap.** Sorting is free because the gap is symmetric, and after sorting each pair's compatibility becomes a value-window `f[i] in [f[j]-R, f[j]-L]` on the smaller element; the symmetric "count both sides per fork" first cut felt natural but a trace returned `16` against the true `8`, exposing that summing over both endpoints of every unordered pair double-counts by construction, with the deceptive `/2` patch hiding a modular-division hazard; the structural fix is to count each pair once at its larger sorted endpoint over the prefix `i < j`, which two monotone pointers `lo, hi` sweep in `O(n)`; the inclusive band forces `<=` on the upper boundary and `<` on the lower (verified on `[0,2,5]` so the gap-exactly-`L` and gap-exactly-`R` pairs both count); and because `total` stays exact in 64-bit with no intermediate modular arithmetic, the single `% MOD` at output closes out the large-count, `L=0`, never-binds, and empty/singleton corners.
