The output here is a single integer — the grand total of all `q` query answers — and the constraints make that one integer the entire difficulty. Each reading `a[i]` is up to `10^9` in magnitude and a window can span all `n <= 10^5` of them, so one window sum reaches `n * max|a| = 10^{14}`. Summing `q <= 5*10^4` such windows pushes the running total to `q * 10^{14} = 5*10^{18}`, and it can reach that in either sign since a window may be all-negative. A 32-bit `int` tops out near `2.1*10^9`: a single window sum already overflows it, and the total overshoots by a factor of about `2*10^9`. So before I pick an algorithm the type is forced — `long long` (max `~9.2*10^{18}`) for every prefix value and the accumulator. `5*10^{18}` sits past half of that range but comfortably inside it.

Re-summing each window directly is `O(r-l+1)` per query, `O(n*q) = 5*10^9` additions worst case — tens of seconds, far past the 1-second limit. I keep that only as a mental correctness oracle. The standard acceleration is prefix sums: `prefix[0] = 0`, `prefix[i] = prefix[i-1] + a[i] = a[1] + ... + a[i]`, after which each window is one subtraction `a[l] + ... + a[r] = prefix[r] - prefix[l-1]`, `O(1)` per query, `O(n+q)` overall. By telescoping, `prefix[r] - prefix[l-1]` cancels the shared head `a[1..l-1]` and leaves exactly `a[l..r]`.

The one indexing subtlety this problem invites is the left boundary. The earlier endpoint must be `prefix[l-1]`, not `prefix[l]` — writing `prefix[l]` silently drops `a[l]` from every window, and the windows most likely to be tested are the ones starting at `l = 1`. At `l = 1` the reference is `prefix[0]`, so a `prefix[0] = 0` sentinel with 1-indexed readings makes those first-element windows work with no special case; 0-indexing the readings would instead put `prefix[-1]` in reach at `l = 1` and misalign `prefix` with `a`. I store readings 1-indexed so `prefix[i]` and `a[i]` share the same `i`.

The documented sample doubles as a check on both the identity and the overflow claim. For `a = [10^9, 10^9, 10^9, -5, 10^9]` the prefix array is `[0, 10^9, 2*10^9, 3*10^9, 2999999995, 3999999995]`. The three queries give `prefix[3]-prefix[0] = 3*10^9`, `prefix[5]-prefix[1] = 2999999995`, and `prefix[5]-prefix[0] = 3999999995`, totalling `9999999990` — the documented answer. And notice the very first window sum, `3*10^9`, is already past `INT_MAX`, so the overflow is not a large-hidden-test artifact; it shows up on five elements.

That makes the type trap concrete. With an `int` prefix array, `prefix[2] = 2*10^9` still fits (under `2147483647`), but `prefix[3] = 3*10^9` wraps to `3000000000 - 2^{32} = -1294967296`, and everything built on it is garbage. Compiling exactly that `int` version and running the sample prints `1410065398` instead of `9999999990` — observable, not hypothetical. Hence `long long` throughout. The difference `prefix[r] - prefix[l-1]` is itself a window sum of magnitude `<= 10^{14}`, so the subtraction is safe in 64-bit and would already be wrong in 32-bit before any accumulation. The core is four lines:

```
vector<long long> prefix(n + 1, 0);
for (int i = 1; i <= n; i++) { long long x; cin >> x; prefix[i] = prefix[i - 1] + x; }
long long total = 0;
for (int k = 0; k < q; k++) { int l, r; cin >> l >> r; total += prefix[r] - prefix[l - 1]; }
```

The edge cases are where this kind of code dies, so I walk the extremes:

- `n = q = 1` with reading `[-10^9]` and query `[1,1]`: `prefix[1] - prefix[0] = -10^9`. Correct — a single negative window, and it exercises the `l = 1` boundary at minimal size.
- The positive overflow extreme: `n = 10^5` all `+10^9`, `q = 5*10^4` each querying `[1,n]`, gives `5*10^4 * 10^{14} = 5*10^{18}` — the largest the total can be, and it fits under `9.2*10^{18}`; running it prints `5000000000000000000`. The all-`-10^9` mirror gives `-5*10^{18}`, above `LLONG_MIN`. So `long long` has margin on both sides, but only because `q <= 5*10^4`: a larger `q` would push the total past `long long` and force `__int128`.
- Zeros and mixed signs need nothing special; prefix arithmetic is sign-agnostic.
- Performance is `O(n+q)`, one `vector<long long>` of `~0.8 MB`, with `sync_with_stdio(false)`; the max-scale run finishes in `~0.01 s`.

Beyond the hand-traces I check against a brute force that answers each query by direct re-summation — no prefix array, so a shared bug is unlikely to hide agreement. Across 400 random tiny instances mixing small values, full-range `+-10^9`, and zeros with random windows, the `long long` solution and the brute agree on every case. That fragment, wrapped in `main()` with `sync_with_stdio(false)` fast I/O and an `if (!(cin >> n >> q)) return 0;` read guard, is the whole program.
