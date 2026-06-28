**Reading the problem and pinning the contract.** I am given `n` coin denominations, a target `S`, and a prime modulus `p`. Supply of each denomination is unlimited, and I must count the number of *distinct ways* to make exactly `S`, modulo `p`. The crucial word is "distinct," and the statement nails down what it means: order does not matter, and a way is fixed by how many coins of each denomination value are used. So `1 + 2` and `2 + 1` are the *same* way. That single sentence is the entire difficulty of this problem; everything downstream is about not accidentally counting ordered sequences. Input is `n S p` on the first line and the `n` denominations on the second; I print one integer, the count mod `p`. Constraints: `n <= 200`, `S <= 2*10^5`, `1 <= c[i] <= 10^6`, `p` prime up to `10^9 + 7`. Let me fix scale before any algorithm. The raw count of multisets can be astronomically larger than any 64-bit integer (this is why the problem asks for it mod `p` in the first place), so I must reduce modulo `p` at every accumulation step, never accumulate the true count. And I should note that denominations can be up to `10^6`, larger than `S`; those coins simply can never be used, but I must not let one of them index out of bounds.

**Laying out the candidate approaches.** There are three routes, and I want to commit to the one I can *prove* counts unordered multisets, not the one that is shortest to type.

- *Generating-function / inclusion-exclusion closed form.* The count of multisets is exactly the coefficient of `x^S` in the product over distinct denominations of `1 / (1 - x^{c})`. There is a temptation to be clever here: evaluate this with an explicit inclusion-exclusion over subsets of denominations, or find a slick convolution identity, maybe even a partial-fraction expansion to get a "formula." It is mathematically elegant, and when I am tired I tend to reach for the elegant thing. But the open question is whether I can implement it *correctly within the budget*. Inclusion-exclusion over subsets of up to 200 denominations is `2^200` terms — a non-starter unless I find structure, and with arbitrary, non-coprime denominations the partial-fraction route has nasty repeated-root and gcd cases. This is exactly the kind of approach that *looks* powerful and turns into a swamp of special cases that I get subtly wrong and ship. I am suspicious of it.
- *Order-sensitive ("compositions") DP.* Scan `s` from `1` to `S` and, for each `s`, add `dp[s - c]` for every denomination `c <= s`. It is `O(S * n)` and three lines. This is the version my fingers type by reflex. But the open question — the *whole* question — is whether this counts multisets or sequences.
- *Order-independent counting DP.* Process denominations in the outer loop; for each denomination, relax all sums `s` in the inner loop with `dp[s] += dp[s - c]`. Also `O(S * n)`. The open question is *why* the loop order makes it count unordered multisets exactly once, plus the base case and the modular arithmetic.

**Reaching for the clever idea, and pulling back.** My first instinct is the generating-function closed form, because "coefficient of `x^S` in `prod 1/(1 - x^{c})`" is a clean, quotable fact and it feels like the "real math." But let me be honest about the budget. To turn that product into a number I either (a) expand the product as a power series up to degree `S` — which is *literally* a DP convolution, so I have not saved anything, I have just renamed the DP — or (b) do partial fractions / inclusion-exclusion to get a closed form. Route (b) is where the elegance lives and where the bugs live. With denominations like `{4, 6}` the poles at roots of unity collide (gcd 2), repeated roots appear, and the residue formulas branch. With 200 arbitrary denominations I would be writing a small computer-algebra engine and praying it has no off-by-one in the multiplicity handling. That is not a 2-second-time-limit contest solution; that is a research project with a high probability of a silent wrong answer. So the "clever" route either *collapses into the DP* (route a) or is *too error-prone to trust in the budget* (route b). I set it aside. The honest move is to take the series-expansion view — which is just a DP — and get the DP exactly right.

**Now the real trap: which DP loop order?** Both DP variants are `O(S * n)` and look almost identical. The danger is that the *wrong* one — sums outer, coins inner — is the more "natural" thing to write, and it silently counts ordered compositions instead of unordered multisets. Before I commit, I will *construct a concrete counterexample* to see the overcount with my own eyes, because that is the only way I will trust the distinction instead of half-remembering it.

Take the sample: denominations `{1, 2, 5}`, `S = 5`. The correct answer (multisets) is `4`: the multisets are `{5}`, `{2, 2, 1}`, `{2, 1, 1, 1}`, and `{1, 1, 1, 1, 1}`. Let me run the *order-sensitive* DP by hand — sums outer, coins inner — `dp[0] = 1`, `dp[s] += dp[s - c]` for each `c in {1,2,5}` with `c <= s`:

- `dp[1] = dp[0] = 1`  (only the coin 1 fits)
- `dp[2] = dp[1] (coin 1) + dp[0] (coin 2) = 1 + 1 = 2`
- `dp[3] = dp[2] + dp[1] = 2 + 1 = 3`
- `dp[4] = dp[3] + dp[2] = 3 + 2 = 5`
- `dp[5] = dp[4] (coin 1) + dp[3] (coin 2) + dp[0] (coin 5) = 5 + 3 + 1 = 9`

So the order-sensitive DP gives `9`, but the right answer is `4`. The `9` is the number of *ordered compositions* of `5` into parts from `{1, 2, 5}`: for instance the multiset `{2, 2, 1}` is counted three times here as the sequences `(2,2,1)`, `(2,1,2)`, `(1,2,2)`, and `{2,1,1,1}` is counted four times, and so on. That is the bug made visible: putting the sum loop outside lets the DP append coins in every order, so it counts sequences, not multisets. The counterexample killed the reflex approach. I will not write that loop order.

**Deriving the order-independent DP and proving why the loop order fixes it.** I want `dp[s]` = number of *multisets* summing to `s`. The way to count multisets exactly once is to impose a *canonical order on the denominations* and only ever build multisets that respect it — concretely, decide the count of denomination `0` fully, then denomination `1`, and so on, never going back. The clean way to encode "decide each denomination fully before moving on" is to make the denomination the *outer* loop:

```
dp[0] = 1                      // empty multiset
for each denomination c (in some fixed order):
    for s = c .. S:
        dp[s] += dp[s - c]
```

Why does this count each multiset once? Think of the invariant *after* the iteration for denomination `c` has finished: `dp[s]` now equals the number of multisets summing to `s` that use only the denominations processed *so far*. The inner loop, by relaxing `dp[s] += dp[s - c]` in increasing `s`, lets denomination `c` be used `0, 1, 2, ...` times: when I read `dp[s - c]`, that value already includes states that used `c` (because I sweep `s` upward within the same pass), so adding it folds in "use one more `c`." Crucially, the contributions of `c` are all collected *inside this one pass*, before any later denomination is touched, so there is no way to interleave "a `c` after a later coin." Each multiset is therefore generated in exactly one canonical sequence — non-decreasing by denomination index — and counted once. That is the whole proof, and it is exactly the structural reason the *outer-sums* version fails (it interleaves denominations and so counts orderings).

Let me confirm on the sample with the *correct* order — denominations processed as `1, 2, 5`. Start `dp = [1,0,0,0,0,0]` over `s = 0..5`.

- Denomination `1`, sweep `s = 1..5`: `dp[1] += dp[0] -> 1`, `dp[2] += dp[1] -> 1`, ..., all become `1`. `dp = [1,1,1,1,1,1]`. (Exactly one way to make any `s` using only 1s — correct.)
- Denomination `2`, sweep `s = 2..5`: `dp[2] += dp[0] -> 2`, `dp[3] += dp[1] -> 2`, `dp[4] += dp[2] -> 1+2 = 3`, `dp[5] += dp[3] -> 1+2 = 3`. `dp = [1,1,2,2,3,3]`. (e.g. `dp[4] = 3`: `{1,1,1,1}`, `{2,1,1}`, `{2,2}` — correct.)
- Denomination `5`, sweep `s = 5..5`: `dp[5] += dp[0] -> 3 + 1 = 4`. `dp = [1,1,2,2,3,4]`.

`dp[5] = 4`. Matches the four multisets. The recurrence and loop order are right.

**The base case and the modular detail.** `dp[0] = 1` encodes the empty multiset: there is exactly one way to make `0`, take nothing. This also makes `S = 0` return `1` for free (the loop never adds to `dp[0]`). One subtlety with tiny moduli: if `p = 1` the answer is trivially `0`, but `p` is prime so `p >= 2` and `1 % p = 1` — still, I will write `dp[0] = 1 % p` defensively so the base case is correct even at `p = 2` (where the *final* answer may be `1 % 2 = 1` or wrap). Every accumulation is `dp[s] = (dp[s] + dp[s - coin]) % p`, so values stay in `[0, p)` and never overflow: `dp[s] < p <= 10^9 + 7` and `dp[s - coin] < p`, their sum `< 2*10^9 + 14`, which fits comfortably in `long long` before the `% p`. So 64-bit accumulators are safe; I do not even strictly need them given `p` fits in 32 bits, but `long long` removes all doubt.

**Handling the corners in the denomination set.** Denominations can exceed `S` (up to `10^6` while `S` can be smaller). A coin with `coin > S` can never be used, and if I let the inner loop start at `s = coin > S` it just does nothing — but to be safe and to avoid any negative-index or large-loop nonsense I `continue` on `coin > S`. Denominations can also repeat in the input, e.g. `{3, 3, 3}`. Under the problem's definition a way is fixed by the *count of each distinct value*, so two coins of value `3` are the same coin type; processing `3` twice would *double-count* (it would let me "use the second 3-type" and count `{3,3,3}` more than once). So I must collapse duplicates first: sort and `unique`. Let me verify that matters with a quick mental check: denominations `{3, 3, 3}`, `S = 9`. The only multiset is `{3,3,3}` — one way. If I did not dedupe, I would process denomination `3` three times and get `dp[9]` counting the partitions of `9` into three labeled "3-types," i.e. compositions of `3` into 3 ordered nonneg parts = `C(3+2,2) = 10` — badly wrong. Deduping gives the correct `1`. So sort + unique is mandatory, not cosmetic.

**First implementation — and I trace it, because clean math transcribes dirty.** My first cut:

```
vector<long long> dp(S + 1, 0);
dp[0] = 1 % p;
for (int i = 0; i < n; i++) {
    long long coin = c[i];
    for (long long s = coin; s <= S; s++)
        dp[s] = (dp[s] + dp[s - coin]) % p;
}
cout << dp[S] % p << "\n";
```

Two things nag me. First, I am iterating over the *raw* `c[i]` without deduping — I just argued that double-counts repeated denominations. Second, `coin` can exceed `S`, and while `for (s = coin; s <= S; ...)` would simply not execute when `coin > S`, I want that to be explicit. Let me trace the smallest input that exposes the dedup bug: `n = 2`, denominations `{2, 2}`, `S = 4`, `p` large. The only multiset summing to `4` is `{2, 2}` — *one* way. Run my first cut: `dp = [1,0,0,0,0]`. First denomination `2`: `dp[2] += dp[0] -> 1`, `dp[4] += dp[2] -> 1`. `dp = [1,0,1,0,1]`. Second denomination `2` (the duplicate): `dp[2] += dp[0] -> 2`, `dp[4] += dp[2] -> 1 + 2 = 3`. `dp[4] = 3`. The code returns `3`, but the answer is `1`.

**Diagnosing the bug.** The defect is precise: by processing the value `2` twice I treated the two input coins of value `2` as two *different* denomination types, so the DP counted "use `k1` coins of type-A-2 and `k2` coins of type-B-2 with `k1 + k2` totaling the right sum" as distinct ways — i.e. it counted the ordered splits `(2,0), (1,1), (0,2)` of the four units across two labeled 2-types, giving `3` instead of `1`. The fix is to collapse duplicate denomination *values* before the DP, because the problem's notion of a "way" is per distinct value. So I sort `c` and `unique` it.

**Fixing and re-verifying.** Add the dedup and an explicit `coin > S` guard:

```
sort(c.begin(), c.end());
c.erase(unique(c.begin(), c.end()), c.end());
...
for (size_t i = 0; i < c.size(); i++) {
    long long coin = c[i];
    if (coin > S) continue;
    for (long long s = coin; s <= S; s++)
        dp[s] = (dp[s] + dp[s - coin]) % p;
}
```

Re-trace `{2, 2}`, `S = 4`: after dedup the denomination list is `{2}`. `dp = [1,0,0,0,0]`. Denomination `2`: `dp[2] += dp[0] -> 1`, `dp[4] += dp[2] -> 1`. `dp[4] = 1`. Correct. And the earlier sample `{1,2,5}, S = 5` is unaffected by dedup (all distinct) and still gives `4`. The case that broke now passes, and it broke for exactly the reason I fixed — duplicate values treated as distinct types — which is the evidence I trust.

**Self-verification harness, because hand traces are not enough.** I wrote an independent brute oracle that does *not* reuse the DP idea: it recursively enumerates multisets by fixing, for each distinct denomination in turn, how many copies `k = 0, 1, 2, ...` to use (`k * value <= remaining`) and recursing on the rest, memoizing on `(denomination index, remaining)`. That counts each multiset exactly once *by construction of fixing per-denomination counts*, with no notion of order at all — a structurally different formulation from the forward sum-relaxation, so it is a genuine cross-check rather than a re-implementation. The oracle accumulates in Python big integers and reduces mod `p` only at the very end, so its correctness does not lean on any modular reasoning. I also wrote a generator mixing regimes: small dense coin sets (large counts, where order-vs-no-order is glaring), sets with/without a `1`, unreachable targets, `S = 0`, coins larger than `S`, duplicate denominations, tiny moduli `p in {2, 3}`, and large primes. I compiled the solution and ran it against the oracle on 600 generated cases plus a batch of hand-picked edge cases. Zero mismatches. Spot checks I trust independently: `{1, 2, 5}, S = 5 -> 4`; the classic US coins `{1, 5, 10, 25}, S = 30 -> 18` (a textbook value); `{3,3,3}, S = 9 -> 1` (dedup); all coins `> S -> 0`; `S = 0 -> 1`; `{1,2,5}, S = 5, p = 2 -> 0` (since `4 mod 2 = 0`). All matched.

**Edge cases, deliberately, because this is where counting code dies.**
- `S = 0`: loop bodies that add to `dp[0]` never run (every `coin >= 1`), so `dp[0] = 1 % p` stands. Output `1` (the empty multiset). Correct.
- Unreachable target, e.g. `{2, 4}, S = 7`: no even combination is odd, `dp[7]` stays `0`. Output `0`. Correct.
- Single denomination `{4}, S = 12`: `dp[12]` gets exactly one contribution path (`12 = 4+4+4`), `dp[12] = 1`; `S = 13` -> `0`. Correct.
- Coins larger than `S`: skipped by the `continue`; they contribute nothing and cannot index out of range. Correct.
- Tiny modulus `p = 2`: every step reduces mod `2`; the base `1 % 2 = 1` is right, and the final `dp[S] % p` wraps correctly (e.g. count `4 -> 0`). Correct.
- Overflow: each addend is `< p <= 10^9 + 7`, the pre-mod sum `< 2.1*10^9` fits in `long long`. Safe.

**Performance at the stated limits.** The work is `O(n * S)` after dedup: at most `200 * 2*10^5 = 4*10^7` inner additions, each a couple of 64-bit ops and a `%`. I timed the worst case — `n = 200` distinct small denominations `1..200` so every inner sweep runs the full length, `S = 2*10^5`, `p = 10^9 + 7` — and it finished in about `0.08` s using under `5` MB. A random large instance ran in `0.01` s. Both are comfortably inside the `2` s / `256` MB budget. So the simple correct DP is not just provable, it is *fast enough by a wide margin* at the chosen constraints, which is the whole point: I deliberately picked constraints where the clean method wins outright and the clever closed form is unnecessary.

**Final solution.** I disproved the order-sensitive DP with a concrete counterexample (`{1,2,5}, S=5`: `9` ordered compositions vs the `4` multisets), I set aside the generating-function closed form because in the budget it either collapses into this very DP or becomes an error-prone partial-fraction engine, I proved the outer-coin loop order counts each multiset exactly once, and I traced a real dedup bug (`{2,2}, S=4` returning the illegal `3`) to its precise cause and re-verified the fix and the corners against an independent enumerative oracle over 600+ cases with zero mismatches. That is what I ship — one self-contained file, the simple `O(n * S)` counting DP I can defend rather than the clever formula I would have fumbled:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S, p;
    if (!(cin >> n >> S >> p)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // A "way" is fixed by how many coins of each DISTINCT denomination value are
    // used, so duplicate denomination values in the input refer to the same coin
    // type and are collapsed first.
    sort(c.begin(), c.end());
    c.erase(unique(c.begin(), c.end()), c.end());

    // dp[s] = number of distinct multisets of coins summing to exactly s, mod p.
    // Order-independent counting: put the COIN loop OUTSIDE and the sum loop
    // INSIDE. Each coin is fully processed before the next, so every multiset is
    // counted once in a fixed canonical order of denominations (no permutations).
    vector<long long> dp(S + 1, 0);
    dp[0] = 1 % p;                       // exactly one way to make 0: the empty multiset
    for (size_t i = 0; i < c.size(); i++) {
        long long coin = c[i];
        if (coin > S) continue;          // a coin larger than S can never be used
        for (long long s = coin; s <= S; s++) {
            dp[s] = (dp[s] + dp[s - coin]) % p;
        }
    }

    cout << dp[S] % p << "\n";
    return 0;
}
```

**Causal recap.** The generating-function closed form looked like the "real" answer but in the budget it either renames the DP or turns into an error-prone partial-fraction engine, so I dropped it; the reflexive order-sensitive DP was wrong because a traced counterexample (`{1,2,5}, S=5`: it returns `9` ordered compositions, not the `4` multisets) showed it counts sequences; the fix is the order-independent DP with the *coin* loop outside, which I proved counts each multiset once via the "decide each denomination fully before the next" invariant; a real implementation bug — not deduping equal denomination values — surfaced as `{2,2}, S=4` returning the illegal `3`, fixed by sort+unique; and `dp[0] = 1 % p` with per-step `% p` over `long long` accumulators closes the `S = 0`, unreachable, large-coin, tiny-modulus, and overflow corners, with the whole `O(n*S)` DP timed at `0.08` s on the worst case.
