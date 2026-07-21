I have `n` beacons, each carrying a 30-bit code `a[0..n-1]`, and I count segmentations of the strip in which every contiguous segment's XOR signature has even popcount, modulo `1000000007`. The scale decides the algorithm class before anything else: `n` reaches `2*10^5`, so the `2^(n-1)` possible segmentations are out of reach for enumeration, and even the natural `O(n^2)` DP — for each prefix length, rescan every earlier cut point — is about `4*10^10` popcount tests, well past a one-second budget. I need `O(n)`. And I pin the predicate down in words first, because it has two convincing impostors: "even popcount of a range XOR" is *not* "the XOR equals zero" and *not* "the XOR value is even". Codes stay below `2^30` and fit a 32-bit int, but the count grows like `2^(n-1)`, so every accumulator is a `long long` reduced mod `1000000007`.

The DP is one-dimensional. Let `dp[i]` be the number of clean segmentations of the first `i` beacons, with `dp[0] = 1` (the empty prefix has the one empty segmentation). Extend by choosing where the *last* segment starts: if it covers beacons `j .. i-1`, everything before `j` is an independently clean segmentation counted by `dp[j]`, and the last segment is admissible exactly when its signature has even popcount. Prefix XOR expresses that signature without recomputation: set `P[0] = 0` and `P[i] = a[0] XOR ... XOR a[i-1]`; the inner terms telescope under XOR, so `sig(j, i-1) = P[i] XOR P[j]`. Hence

```
dp[i] = sum over j in [0, i-1] of dp[j]   for which   popcount(P[i] XOR P[j]) is even.
```

Literal, that is `O(n^2)`. To collapse it to `O(n)` the popcount test must depend on `P[j]` alone, so the eligible `dp[j]` can sit in a running bucket keyed by that property.

The key is a popcount identity: `popcount(x XOR y)` is even iff `popcount(x)` and `popcount(y)` share the same parity. To see it, XOR keeps exactly the bits where `x` and `y` differ; every bit they share (in `x AND y`) is counted in both `popcount(x)` and `popcount(y)` but absent from `x XOR y`, so it shifts `popcount(x) + popcount(y)` by 2 — zero mod 2 — relative to `popcount(x XOR y)`. Thus `popcount(x XOR y) ≡ popcount(x) + popcount(y) (mod 2)`, and mod-2 addition is XOR. So the eligibility of `dp[j]` reduces to a single bit: `parity(popcount(P[j]))` must equal `parity(popcount(P[i]))`.

Two impostors sit right next to this identity, close enough that either could pass a glance, so I settle each against concrete numbers. The first reads "even popcount" as "even value" and buckets on the low bit `P & 1`. It comes apart at the smallest inputs: `3 = 11` has popcount 2 (even) yet is odd, `5 = 101` the same, `2 = 10` has popcount 1 (odd) yet is even — a genuinely different predicate. End to end it corrupts the count: on `a = [0, 0, 0, 2]`, whose full-strip signature `2` has odd popcount so *no* segmentation is clean and the answer is `0`, the LSB key sees every prefix as even and doubles at each step to `8`. The second impostor is the non-parity form `popcount(x XOR y) = popcount(x) XOR popcount(y)`, bit-XORing the counts themselves; testing random pairs shows it disagrees on roughly 94% of them, while the parity reduction never fails. Only the reduction to a single parity bit survives.

So define `key[j] = parity(popcount(P[j]))`, and the transition becomes `dp[i] = sum of dp[j]` over `j < i` with `key[j] == key[i]`. Keep two running sums, `bucket[0]` and `bucket[1]`, each the total `dp[j]` seen so far under that parity. At step `i`, read `dp[i] = bucket[key[i]]`, then fold `dp[i]` back into `bucket[key[i]]` so later steps see it. The base `dp[0] = 1` must be registered first under `key[0] = parity(popcount(0)) = 0`. Nothing needs to be stored as an array — `P` and the two buckets roll forward in `O(n)` time, `O(1)` space.

First cut of the loop:

```
long long bucket[2] = {0, 0};
int P = 0;
long long dp = 1;                       // dp[0]
for (int i = 1; i <= n; i++) {
    int x; cin >> x;
    P ^= x;
    int par = __builtin_parity((unsigned)P);
    dp = bucket[par] % MOD;
    bucket[par] = (bucket[par] + dp) % MOD;
}
cout << dp % MOD << "\n";
```

Trace it on the sample `a = [3, 1, 2]`, answer `2`. Prefixes `P = [0, 3, 2, 0]`; popcount parities `[0, 0, 1, 0]`. With `bucket = [0, 0]`: `i=1` (`par=0`) reads `bucket[0]=0`, writes `0`; `i=2` (`par=1`) reads `0`; `i=3` (`par=0`) reads `0`. It prints `0`, not `2`. The bug is that `dp[0] = 1` never entered the buckets — I held it in the local `dp` but never did `bucket[key[0]] += dp[0]` before the loop, so when `i=1` reads `bucket[0]` the seed is missing and every `dp[i]` collapses to `0`. The buckets must accumulate `dp[j]` for *all* `j` from `0` upward, and `j = 0` is one of them. Seed it before the loop, while `P` is still `P[0] = 0`:

```
bucket[__builtin_parity((unsigned)P)] = 1;   // bucket[0] = 1
```

Re-trace: `bucket = [1, 0]`. `i=1` (`par=0`): `dp = bucket[0] = 1`, `bucket[0] = 2`. `i=2` (`par=1`): `dp = 0`. `i=3` (`par=0`): `dp = bucket[0] = 2`. Prints `2`. It matches the enumeration — the two clean segmentations are `[3,1,2]` (signature `0`, popcount 0) and `[3] | [1,2]` (signatures `3` and `3`, popcount 2 each); the other two carry a segment of odd popcount.

Corner behavior. `n = 1` with `a = [1]`: `P = [0, 1]`, keys `[0, 1]`, so `dp = bucket[1] = 0` — the answer is legitimately `0`, since the lone segment `[1]` has odd-popcount signature and no segmentation cleans it. Empty input stream: `cin >> n` fails, `return 0`, nothing printed, matching the contract; a present `n = 0` runs the loop zero times and prints `dp = 1`, the empty segmentation. Overflow: `dp` and both buckets are `long long` reduced mod `1000000007` after each update, so they never exceed about `2*10^9` even though the true count can approach `2^(n-1)` — which is why the answer is reported modulo the prime. Performance: one pass, two adds and a `__builtin_parity` per beacon, so `n = 2*10^5` finishes in a few kilobytes and well under the limit.

The final loop is the seeded `O(n)` pass above; the full program is in the answer.
