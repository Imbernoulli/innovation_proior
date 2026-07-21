The pivotal word in this problem is *distinct*. I have a lowercase string `s` of length `n` and an integer `k`, and I count the length-`k` substrings that occur at two or more starting positions — but a substring is its content, not its location, so a string readable at three indices contributes `1` to the answer, not `3` and not `2`. That word alone decides the counting step, and it is where this kind of problem quietly goes wrong. First the scale: `n` and `k` are each up to `2*10^5`, and crucially `k` is *not* bounded by `n` — it may exceed `n`, be `0`, or equal `n` exactly. There are `n - k + 1` windows when `1 <= k <= n` and none otherwise, so the answer never exceeds `n - k + 1`, comfortably in 32 bits; I carry it in `long long` anyway to mix cleanly with `n - k + 1`. Input is `n k` then the string on the next line; output is one integer.

The definitional approach — build all `n - k + 1` windows as `std::string`, sort, sweep counting runs of length `>= 2` — is obviously correct, but every compare touches up to `k` characters, so it is `O(n*k*log n)`; at `n = 2*10^5`, `k ~ n/2` that is on the order of `10^{11}` comparisons, hopeless in a second. I keep it as a brute-force oracle to test against, not as the submission. The affordable route is a polynomial rolling hash: reduce each window to an integer fingerprint updated in `O(1)` per shift, `O(n)` for all of them, then sort the `n - k + 1` fingerprints and count equal groups in `O(n log n)`. The asymptotics are fine; the danger is three small places where this code goes subtly wrong — the exact roll step, the window count, and the map from a group of equal fingerprints to the count the problem actually asks for.

Fingerprint a window `c_0 ... c_{k-1}` (each `c_t` a small positive integer for its letter) by Horner's rule, `h = c_0 * B^{k-1} + c_1 * B^{k-2} + ... + c_{k-1}  (mod M)`. Sliding right by one drops `c_0` and appends `c_k`; the leading term of `h` is `c_0 * B^{k-1}`, so I subtract that, multiply the rest by `B` to push every remaining term up one power, and add the new character:

`h' = (h - c_0 * B^{k-1}) * B + c_k   (mod M).`

The weight I remove is `c_0 * B^{k-1}`, **not** `c_0 * B^k` — this is exactly where off-by-ones live, so I need `B^{k-1} mod M`, not `B^k`. A single ~`10^9`-modulus hash would invite a birthday collision over `2*10^5` windows, merging two different substrings into one group and corrupting the count, so I use two independent moduli/bases and pack `(h1, h2)` into a 64-bit key `(h1 << 32) ^ h2`, an effective space near `10^{18}`; equal real substrings always give identical pairs, so true repeats are never split.

On the sample `s = "ababbaba"`, `k = 3`, the windows at indices 0..5 are `aba, bab, abb, bba, bab, aba`; by content, `aba -> {0,5}`, `bab -> {1,4}`, `abb -> {2}`, `bba -> {3}`. Two groups have size `>= 2`, so the answer is `2` — the number of *groups* of size `>= 2`, not the number of repeated occurrences (which is 4) nor extra occurrences (2 here, by coincidence). That distinction between groups and occurrences is exactly what the counting step below must get right, and the sample's `2` is what I will re-check the sweep against.

The leading-power warning is not idle: it is easy to write the roll as removing `out * B^k`, and the smallest rolling input exposes it. Take `n=2, k=1, s="aa"`: two single-character `"a"` windows, so the answer must be `1`. With the wrong `p = B^k = B`, the first window gives `h1 = 1`; the roll computes `h1 = (1 + M1 - (1*131)%M1)%M1 = M1 - 130`, then `*131 + 1`, a large nonzero value — two identical strings get different fingerprints, the grouping sees two singletons, and the program prints `0`. The tell is that only rolling windows are corrupted while single-window cases stay correct — the signature of a wrong leading power rather than a hash fluke. The fix is `p = B^{k-1}` (precompute loop `k-1` times, not `k`). Re-tracing `"aa", k=1` with `p=1`: the roll gives `h1 = (1 + M1 - 1)%M1 = 0`, then `0*131 + 1 = 1`, matching the first window — one group of size 2, answer `1`.

With fingerprints correct, the *distinct-vs-occurrences* trap remains, and it is the reason the problem exists. The tempting sweep marks every window that has an equal neighbour:

```
sort(keys.begin(), keys.end());
for (i = 0; i < m; i++)
    if ((i > 0 && keys[i]==keys[i-1]) || (i+1 < m && keys[i]==keys[i+1]))
        ans++;
```

On the sample this marks both `aba`s and both `bab`s, giving `ans = 4` against the correct `2`. It is counting repeated *occurrences*: a substring appearing `t >= 2` times adds `t`, not `1`. The unit is wrong. The correct sweep walks each maximal run of equal keys once and adds `1` iff the run length is `>= 2`:

```
long long ans = 0, i = 0;
while (i < m) {
    long long j = i;
    while (j < m && keys[j] == keys[i]) j++;   // [i, j) is one group
    if (j - i >= 2) ans++;                      // one distinct substring, once
    i = j;
}
```

Now the runs `[aba,aba]` and `[bab,bab]` each add 1 and the two singletons add 0, giving `2`; on `"aaaa", k=1` the single run of length 4 adds 1 (the one distinct repeated `"a"`), where the buggy sweep would have said 4.

The `k`/`n` boundaries are the last place this dies, and each maps onto a line of the guard. `k = 0` and `k > n` leave no window — `n - k + 1` would be meaningless — so `if (k < 1 || k > n) print 0`. `k = n` gives exactly one window, whose group has size 1, answer `0`. `n = 0` must not do an unconditional `cin >> s`, or the stream blocks on the missing token; reading `s` only when `n > 0` keeps parsing clean and the guard then fires. `k = 1` degenerates correctly because `p = B^0 = 1`. Checking against the brute oracle on random small cases — tiny alphabets, `k` spanning `0..n+2` — agrees on every one, including dense-repeat strings over `{a,b}` where groups are large.

The arithmetic stays in range: each modular value is below `2^30`, the largest intermediate is a product of two of them, below `2^60`, fitting in `unsigned long long` before the `% M`; and `h1 << 32 < 2^62` with `h2` occupying the low 32 bits, so the packed halves never overlap and equal pairs give equal keys. Rolling is `O(n)`, the sort of `<= 2*10^5` 64-bit keys `O(n log n)`, the sweep `O(n)` — about `0.01 s` and a few MB on the worst constructed case, well inside `1 s / 256 MB`.
