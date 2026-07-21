I'm computing an echo score: slide a length-`L` window over a length-`n` stream, group windows by identical content, and sum `C(c_w, 2)` over the groups. Two numbers set the whole shape of the solution. First, `L` ranges up to `2*10^5`, the same order as `n`, so anything that touches `L` characters per window — materializing substrings and comparing them — is `O(nL)` and dies: at `L = 10^5`, `n = 2*10^5` that is tens of gigabytes of stored substrings and `~2*10^{10}` character-touches. Second, a single content can occur `~n` times: at `L = 1` on `"aaa...a"` every one of the `n` windows is the same character, so `c_w = n` and `C(n, 2) = n(n-1)/2`, which for `n = 2*10^5` is `19,999,900,000 ≈ 2*10^{10}` — an order of magnitude past the `~2.1*10^9` signed-32-bit ceiling, from *one* group alone. So before I pick an algorithm, two things are fixed: the final accumulator is 64-bit, and — the part that will actually bite — every product that feeds it must be *formed* in 64-bit, not merely stored there.

The substring sort (build all `m` windows as `std::string`, sort, count equal runs) is correct by definition, and I keep it as a brute-force oracle, but its `O(nL log n)` time and `O(nL)` memory rule it out as the shipped solution. What I want instead is a fingerprint per window computable in `O(1)`, so grouping becomes independent of `L`: polynomial hashing gives `O(n log n)` time and `O(n)` memory regardless of `L`.

For the rolling hash I use prefix hashes `h[0] = 0`, `h[i+1] = (h[i]*B + val(s[i])) % M` and powers `p[0] = 1`, `p[i+1] = (p[i]*B) % M`, so `h[i]` is the base-`B` value of the prefix `s[0..i)` with the first character most significant. The hash of the window `s[l..l+L)` is then

```
hash = (h[l+L] - h[l]*p[L]) mod M
```

Subtraction under a modulus can go negative, so I add `M` first: `(h[l+L] + M - (h[l]*p[L]) % M) % M`. Characters map to `val = (unsigned char)c + 1` rather than `0`-based: with the smallest character at value `0`, the strings `"a"`, `"aa"`, `"aaa"` would all hash to `0` and collapse into a single group. The `+1` offset removes that degeneracy.

One modulus near `10^9` is not safe. By the birthday bound the expected number of colliding pairs among `m ≈ 2*10^5` windows is `~C(m,2)/M ≈ 2*10^{10} / 10^9 ≈ 20`, and each accidental collision merges two genuinely different contents and corrupts a count. Two independent moduli and bases, with windows equal only when *both* fingerprints agree, push the effective modulus to `~10^{18}` and the expected false-collision pairs to `~10^{-8}` — negligible, and robust against an adversary who knows one modulus. Each residue sits below `2^30`, so I pack the pair into one 64-bit key as `(x1 << 32) ^ x2` and sort plain `unsigned long long`s with the pair recoverable.

Grouping is then: fingerprint the `m` windows, sort the keys, and for each maximal run of equal keys of length `c` add `C(c, 2)`. The overflow trap lives in that last step, and it is subtle. Written the natural way,

```
int c = j - i;
answer += c * (c - 1) / 2;   // answer is long long
```

*looks* safe because `answer` is 64-bit — but `c` is `int`, so `c*(c-1)` is an `int * int` product formed in 32-bit *before* it is widened for the accumulator. On the overflow witness `c = 200000`: the true `c*(c-1) = 39,999,800,000` wraps modulo `2^32 = 4,294,967,296` to `39,999,800,000 - 9*4,294,967,296 = 1,345,094,336`, and `/2 = 672,547,168`, so `answer` lands on `672,547,168` instead of `19,999,900,000`. The 64-bit accumulator never sees the true product. And it is silent: any small case has `c ≤ 3`, product `≤ 6`, so every generated test passes and only `n` near `2*10^5` exposes it. The fix is to form the product in 64-bit by making the group size a `long long`:

```
long long c = j - i;
answer += c * (c - 1) / 2;   // now a 64-bit multiply
```

`c = 200000LL` gives `39,999,800,000` with no wrap (far under `~9.2*10^{18}`), `/2 = 19,999,900,000`. The type has to be 64-bit at the point of multiplication, not merely at storage.

The other easy miss is the window count itself: the number of length-`L` windows is `n - L + 1`, with starts ranging `0 .. n-L` inclusive. Using `n - L` drops the last window — on the sample it would fingerprint only `ab, ba, ab, ba, aa`, report `ab`×2, and give `1 + 1 + 0 = 2` instead of `4`. With `m = n - L + 1` the largest start is `n - L`, so `getHash` reads `h[n]` and `p[L]`, both in range (`h` has size `n+1`, `L ≤ n`) — no out-of-bounds.

The `L > n` case needs a guard *before* allocation: with no windows `m = n - L + 1 ≤ 0`, and `vector<...> keys(m)` on a negative size cast to `size_t` would try a huge allocation and crash, so I return `0` up front (also covering `L ≤ 0` defensively). The rest fall out of the same machinery: `L = n` is one window contributing `C(1,2) = 0`; `n = 1` likewise `0`; all-distinct windows are singletons summing to `0`; the all-identical and short-period inputs are the overflow witness already handled. Whitespace-skipping `cin >> n >> L` and `cin >> s` parse the two-line input format-agnostically.

For confidence in the grouping I run the hashing solution against the substring-dictionary brute force — a completely different mechanism, actual substrings as dictionary keys — on 600 random small cases: short strings, alphabets of size 1 to 3 to force many equal windows, `L` swept from 1 to past `n` to exercise the empty-window branch. Zero mismatches, and the overflow witness matches `19,999,900,000` by hand. Agreement across an independent oracle is what tells me the grouping is faithful and the per-group arithmetic is right.

The full single-file program — the `L > n` guard, the two prefix-hash tables and powers, `getHash` over `[l, l+L)`, the key sort, and the run scan with the 64-bit group size — is the shipped answer.
