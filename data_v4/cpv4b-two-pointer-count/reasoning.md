The output is a *count* reduced modulo `1 000 000 007`, and that shapes the types before any algorithm does. With `n` up to `10^6` the number of compatible pairs can approach `C(10^6, 2) ~ 5*10^11` — comfortably inside a 64-bit `long long` (ceiling `~9.2*10^18`), so I keep the running count exact and reduce mod the prime once, at the very end; there is no reason to touch modular arithmetic mid-loop. The subtler numeric trap is on the *inputs*: frequencies reach `±10^9` and the band bounds reach `2*10^9`, so a derived window endpoint like `f[j] - R` can sink to `-10^9 - 2*10^9 = -3*10^9`, well outside a 32-bit `int`. Every frequency, band bound, and window endpoint therefore has to be `long long` — an `int` here is a silent wrong answer on the wide-band tests, not a crash.

Quadratic enumeration — test all `i < j` for `L <= |f[i]-f[j]| <= R` — is my definition made literal and my correctness oracle, but `5*10^11` comparisons blow the 2-second budget, so it can only be the brute force I check against. The workable route is to sort and sweep. The gap `|f[i]-f[j]|` is symmetric, so sorting the frequencies changes no pair's compatibility. After sorting (`f[0] <= ... <= f[n-1]`), take positions `i < j`, so `f[i] <= f[j]` and the gap is `f[j]-f[i]`; compatibility `L <= f[j]-f[i] <= R` rearranges into a pure constraint on `f[i]`:

- `f[j]-f[i] >= L`  iff  `f[i] <= f[j]-L`;
- `f[j]-f[i] <= R`  iff  `f[i] >= f[j]-R`.

Fixing the larger position `j`, its compatible partners are the indices `i < j` whose value lies in the closed window `[f[j]-R, f[j]-L]`. Sorted-ness makes that window a contiguous run, and restricting to `i < j` clips it to the prefix. As `j` moves right `f[j]` is nondecreasing, so both window ends are nondecreasing and two monotone pointers sweep the prefix once — `O(n log n)`, dominated by the sort.

The one thing that genuinely bites here is the counting discipline, and the natural framing gets it wrong. My first instinct is to treat each fork as a band center and count partners on *both* sides — lower neighbours in `[f[j]-R, f[j]-L]` and upper neighbours in `[f[j]+L, f[j]+R]` — summed over the whole array. But because `|f[i]-f[j]|` is symmetric, that sees every unordered pair twice, once from each endpoint: the pair `{1,4}` is caught at `f=1` looking up and again at `f=4` looking down. On the sample it returns `16`, exactly `2*8`. Dividing by two would "work", but it is a bad habit under a modulus — it tempts reducing before halving, which destroys the parity, and it drags in a needless modular inverse. The structural fix costs nothing: count each pair once at its *larger* sorted endpoint, i.e. only over the prefix `i < j`. That deletes the upper-side term entirely.

So I keep two pointers over the prefix: `hi` counts prefix elements with value `<= f[j]-L`, `lo` counts those with value `< f[j]-R`, and the partners of `j` number `hi - lo`. Both are clamped to `< j` so they never look past the prefix. The `<=`/`<` asymmetry is the load-bearing detail. The band is closed, so the partner at gap exactly `L` (value `f[j]-L`) and the partner at gap exactly `R` (value `f[j]-R`) must both count — `hi` with `<=` includes `f[j]-L`, and `lo` with `<` leaves `f[j]-R` inside the `[lo, hi)` block.

Tracing the sorted sample `[1,4,5,8,10,13]`, `L=2`, `R=5`, watching `lo,hi`:

- `j=1` (`f=4`): prefix `<= 2`: `1` -> `hi=1`; prefix `< -1`: none -> `lo=0`; add `1`.
- `j=2` (`f=5`): prefix `<= 3`: `4` fails -> `hi=1`; `< 0`: none -> `lo=0`; add `1`.
- `j=3` (`f=8`): prefix `<= 6`: `1,4,5` -> `hi=3`; `< 3`: `1` -> `lo=1`; add `2`.
- `j=4` (`f=10`): prefix `<= 8`: adds `8` -> `hi=4`; `< 5`: adds `4` -> `lo=2`; add `2`.
- `j=5` (`f=13`): prefix `<= 11`: adds `10` -> `hi=5`; `< 8`: adds `5` -> `lo=3`; add `2`.

Total `0+1+1+2+2+2 = 8`, matching the eight compatible pairs `(1,4),(1,5),(4,8),(5,8),(5,10),(8,10),(8,13),(10,13)` — and `8`, not the `16` the both-sides framing gave.

The corners fall out of the same code. `n=0` and `n=1` never add anything — the inner `while`s see `hi<j`/`lo<j` false immediately — so `0`. `L=0` makes the upper bound `<= f[j]`, so equal-frequency forks count each other: `f=[5,5,5]` gives `C(3,2)=3`. When `R` exceeds every achievable gap, `f[j]-R` stays hugely negative, `lo` never leaves `0`, and the lower bound alone governs. `L=0` with a large `R` counts all `C(n,2)` pairs, the densest case; `total` stays under `~5*10^11`, so the single `% MOD` at output involves no subtraction or division under the modulus — which is exactly what dropping the `/2` bought.

That is the shipped program: read `n, L, R` and the frequencies as `long long`, sort, sweep with `lo, hi` over the prefix, and print `total % MOD`; the full module is in the answer.
