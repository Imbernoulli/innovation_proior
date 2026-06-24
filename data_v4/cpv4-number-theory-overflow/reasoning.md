**Reading the problem and pinning the contract.** I have `n` oscillators with integer frequencies `a[0..n-1]`, a modulus `m`, and a target residue `t` with `0 <= t < m`. I must count the unordered pairs `{i, j}`, `i != j`, for which `(a[i] + a[j]) mod m == t`. Input on stdin is `n m t` then the `n` frequencies; I print one integer. Before I pick an algorithm I fix the scale, because here the scale *is* the trap: `n <= 2*10^5`, so the number of unordered pairs is at most `C(n, 2) = n*(n-1)/2 ≈ 2*10^10`. That is roughly ten times `INT_MAX ≈ 2.147*10^9`. So the final answer does not fit in a 32-bit `int`, full stop. And the frequencies themselves go up to `10^9`, which fits in `int` individually but whose sum `a[i] + a[j]` reaches `2*10^9` — already brushing the signed 32-bit ceiling. I write that observation down now and will return to it: every quantity that *counts pairs* or *sums two frequencies* must be 64-bit. An `int` anywhere on those paths is a silent wrong-answer on the large tests, not a crash. This is the decision that the whole problem is really testing.

**Laying out candidate approaches.** Two routes, and I want the one I can both prove correct and run in time.

- *Brute force over pairs.* Two nested loops, test `(a[i] + a[j]) % m == t` for every `i < j`. Obviously correct, `O(n^2)`. At `n = 2*10^5` that is `2*10^10` modulo operations — about two orders of magnitude past what fits in a 1-second limit. Correct but far too slow; I will keep it only as my mental oracle for verifying the fast version.
- *Residue bucketing.* The predicate `(a[i] + a[j]) mod m == t` depends only on `a[i] mod m` and `a[j] mod m`. So I reduce every frequency to its residue, count how many frequencies fall in each residue class, and then count *residue pairs* that complete to `t`. There are at most `m` residues, and `m <= 10^6`, so a single pass over residues is `O(n + m)`. This is the approach to commit to — but the counting of pairs from bucket sizes is exactly where off-by-one and double-counting bugs breed, so I will derive it carefully and trace it.

**Deriving the bucket count.** Let `cnt[r]` be the number of indices `i` with `a[i] mod m == r`, for `r in 0..m-1`. A pair `{i, j}` is resonant iff its residues `(r, s)` satisfy `(r + s) mod m == t`. For a fixed `r`, the residue `s` that completes it is forced: `s = (t - r) mod m`, taken in `0..m-1`. Now I have to count *index pairs*, not residue pairs, and there are two genuinely different cases:

- **Distinct residues, `r != s`.** Every frequency in bucket `r` pairs with every frequency in bucket `s`, and all those `cnt[r] * cnt[s]` index pairs are distinct and resonant. But the unordered residue pair `{r, s}` will be visited twice as the loop runs — once when the loop variable equals `r` (completing to `s`) and once when it equals `s` (completing to `r`). I must add `cnt[r] * cnt[s]` exactly once.
- **Self-paired residue, `r == s`.** This happens when `2r ≡ t (mod m)`. Then both elements of the pair come from the *same* bucket, and I am choosing 2 distinct indices out of `cnt[r]`, which is `C(cnt[r], 2) = cnt[r] * (cnt[r] - 1) / 2`. Crucially it is *not* `cnt[r] * cnt[r]`, because an index cannot pair with itself and order does not matter.

So the clean way to avoid the double count is: as `r` runs `0..m-1`, compute `s = (t - r) mod m`; add `cnt[r] * cnt[s]` only when `r < s`, add `C(cnt[r], 2)` when `r == s`, and skip `r > s` entirely (it was handled when the loop variable was the smaller of the two). That visits each unordered residue pair exactly once.

**Sanity-checking the derivation on the sample.** The sample: `n = 6`, `m = 5`, `t = 3`, `a = [7, 1, 4, 6, 9, 2]`. Reduce mod 5: `7→2`, `1→1`, `4→4`, `6→1`, `9→4`, `2→2`. So the buckets are `cnt[1] = 2` (the `1` and `6`), `cnt[2] = 2` (the `7` and `2`), `cnt[4] = 2` (the `4` and `9`); all other `cnt[r] = 0`. I want residue pairs summing to `3 (mod 5)`. Enumerate: `1 + 2 = 3` ✓ (distinct), `4 + 4 = 8 ≡ 3` ✓ (self), `0 + 3`, `3 + 0` — but `cnt[0] = cnt[3] = 0`, contribute nothing. So the answer is `cnt[1] * cnt[2] + C(cnt[4], 2) = 2*2 + 2*1/2 = 4 + 1 = 5`. The expected output is `5`. The derivation reproduces the sample, which is the evidence that the case split is right before I write a line of the loop.

**First implementation — and immediately a trace, because counting code lies.** My first cut of the counting loop, written quickly:

```
long long answer = 0;
for (long long r = 0; r < m; r++) {
    long long s = ((t - r) % m + m) % m;
    answer += cnt[r] * cnt[s];     // count pairs (r, s)
}
```

This looks innocent. I distrust it precisely because it does not mention the self-pair case or the double-count guard at all — it just multiplies and sums. Let me trace it on the smallest input that exercises both dangers at once: `n = 3`, `m = 6`, `t = 0`, `a = [6, 12, 18]`. All three reduce to `0 (mod 6)`, so `cnt[0] = 3`, everything else `0`. The true answer is the number of unordered pairs among three identical-residue elements, all of which sum to `0 ≡ t`: that is `C(3, 2) = 3`. Now run my loop. For `r = 0`: `s = (0 - 0) mod 6 = 0`, so `answer += cnt[0] * cnt[0] = 3 * 3 = 9`. For `r = 1..5`: `s = (0 - r) mod 6 = 6 - r`, and `cnt[r] = 0` for all of them, so they add nothing. Final `answer = 9`.

**Diagnosing the first bug.** The code returns `9`; the truth is `3`. Two distinct errors are tangled here, and the trace separates them. First, the self-pair: when `r == s` I computed `cnt[r] * cnt[r] = 9`, which counts *ordered* pairs *including each index with itself* — it counts `(i, i)` (3 of those) and counts `(i, j)` and `(j, i)` separately. The correct count of unordered distinct pairs is `C(3, 2) = 3`. Second, even for distinct residues this loop would double count: if there were a nonzero `cnt[1]` and `cnt[5]` pairing to `t = 0`, I would add `cnt[1]*cnt[5]` at `r = 1` and again `cnt[5]*cnt[1]` at `r = 5`. So the loop both mishandles the diagonal and double-counts the off-diagonal. The derivation already told me the fix; my first implementation simply ignored it. I rewrite with the `r < s` / `r == s` / skip-`r > s` split exactly as derived.

**Fix and re-verification of the counting.** The corrected loop:

```
long long answer = 0;
for (long long r = 0; r < m; r++) {
    long long s = ((t - r) % m + m) % m;
    if (r < s)      answer += cnt[r] * cnt[s];
    else if (r == s) answer += cnt[r] * (cnt[r] - 1) / 2;
    // r > s: already handled when the loop variable was s
}
```

Re-trace `a = [6, 12, 18]`, `m = 6`, `t = 0`: `cnt[0] = 3`. At `r = 0`: `s = 0`, `r == s`, add `3 * 2 / 2 = 3`. All other `r` have `cnt[r] = 0`. Final `answer = 3`. Correct. Now re-trace a distinct-residue case to confirm no double count: `n = 4`, `m = 5`, `t = 3`, residues `cnt[1] = 2`, `cnt[2] = 2` (so frequencies like `1, 6` and `2, 7`). At `r = 1`: `s = (3 - 1) mod 5 = 2`, `r < s`, add `cnt[1]*cnt[2] = 4`. At `r = 2`: `s = (3 - 2) mod 5 = 1`, now `r = 2 > s = 1`, skip. Final `4`. The four index pairs are exactly `{1,2},{1,7},{6,2},{6,7}`, all summing to `3 (mod 5)` — correct, and counted once. The two cases that broke before now pass for the reason I fixed.

**Second trace — the overflow I promised myself I would not forget.** The counting logic is right, but I flagged at the very start that the magnitudes are dangerous, so I now stress the *types* deliberately. Construct the worst case: `n = 2*10^5`, `m = 2`, `t = 0`, and every frequency even, say all equal to `4`. Then every frequency reduces to residue `0`, so `cnt[0] = 200000`, and every pair is resonant. The true answer is `C(200000, 2) = 200000 * 199999 / 2 = 19,999,900,000 ≈ 2*10^10`. Suppose for the moment I had declared `cnt` as `vector<int>` and `answer` as `int`, which is the natural first instinct. The self-pair term is `cnt[0] * (cnt[0] - 1) / 2`. With `int cnt`, the multiplication `cnt[0] * (cnt[0] - 1)` is `200000 * 199999 = 39,999,800,000` computed *in `int` arithmetic* — that wraps modulo `2^32` long before the `/ 2` ever runs, and the result is garbage. Even if I had used 64-bit `cnt` but `int answer`, the assignment of `~2*10^10` into a 32-bit accumulator truncates. I actually compiled the `int` version against this exact input and it printed `672547168`, while the correct value is `19,999,900,000`. That is the silent wrong-answer the constraints were engineered to produce: no crash, no warning, just a wrong number on the big hidden test.

**Diagnosing and fixing the overflow.** The fix has to cover *every* arithmetic site that can grow past 32 bits, because a single `int` on the path re-introduces the wrap. There are three such sites: (1) the bucket counts `cnt[r]`, which top out at `n = 2*10^5` — that one actually fits in `int`, but I make it `long long` anyway so that the *products built from it* are evaluated in 64-bit; (2) the per-bucket product `cnt[r] * cnt[s]` and the self term `cnt[r] * (cnt[r] - 1)`, which reach `~4*10^10`; and (3) the running `answer`, which reaches `~2*10^10`. The decisive subtlety is that in C++ the multiplication `cnt[r] * cnt[s]` is performed in the operands' type *before* it is added to `answer` — so making only `answer` a `long long` while leaving `cnt` as `int` would still overflow inside the product. The robust choice is to make `cnt` itself `vector<long long>`; then every product and the accumulation are all 64-bit. With `long long` throughout, the worst case `~2*10^10` sits comfortably inside the `~9.2*10^18` range of signed 64-bit, with enormous headroom. I re-ran the all-even `n = 2*10^5` case with the `long long` version and it prints `19999900000` — the true value. The overflow is closed.

**One more arithmetic corner — the modulo on the target.** I compute `s = (t - r) mod m`. Since `t` and `r` are both in `0..m-1`, `t - r` ranges over `-(m-1) .. (m-1)`, which can be negative. In C++ the `%` of a negative numerator is negative or zero, so a bare `(t - r) % m` could yield, e.g., `-2`, and indexing `cnt[-2]` is undefined behavior. I guard it the standard way: `s = ((t - r) % m + m) % m`, which lands `s` in `0..m-1` regardless of sign. I trace it: `t = 1`, `r = 3`, `m = 5`: `(1 - 3) % 5 = -2`, `+5 = 3`, `% 5 = 3`. And indeed `3 + 3 = 6 ≡ 1 (mod 5)`, so `s = 3` is the correct completing residue. I apply the same `((x % m) + m) % m` form when reducing each frequency, which is harmless here because `a[i] >= 0` but makes the reduction sign-agnostic and obviously correct.

**Edge cases, deliberately, because this is where counting code dies.**
- `n = 0`: the read loop never runs, all `cnt[r] = 0`, the answer loop adds only zeros, output `0`. No pairs exist — correct. (I read `n` first; if input is empty the guarded `cin >> n >> m >> t` fails and I `return 0`, printing nothing, which is the documented "no input" behavior.)
- `n = 1`: one bucket has count `1`, every term is either `cnt[r]*cnt[s]` with one factor `0` or the self term `1*(1-1)/2 = 0`. Output `0` — a single oscillator forms no pair. Correct.
- `m = 1`: there is exactly one residue, `0`, and `t = 0` is forced by `0 <= t < m`. Then `cnt[0] = n` and the only term is the self term `C(n, 2)` — every pair is resonant because every sum is `≡ 0 (mod 1)`. For `n = 4` I get `4*3/2 = 6`; checking by hand, all `C(4,2) = 6` pairs qualify. Correct. This is also a big-answer case: at `n = 2*10^5` it gives `~2*10^10`, again exercising 64-bit.
- Self-paired residue with even count vs odd count: `C(k, 2) = k(k-1)/2` is always an integer because `k(k-1)` is even, so the integer division never loses anything regardless of parity. For `k = 2`: `1`; `k = 3`: `3`; `k = 1`: `0`. Correct, no rounding hazard.
- Maximum frequencies: `a[i] = 10^9` reduces fine under `%`; the residue is in `0..m-1`. The raw sum `a[i] + a[j]` never appears in my fast solution (I work entirely with residues and counts), so the `2*10^9` intermediate that would threaten an `int` in a brute force is sidestepped entirely. Good.
- Overflow, restated: every count, product, and accumulator is `long long`; the maximum value `~4*10^10` (the unsimplified product) and `~2*10^10` (the answer) both fit with three orders of magnitude to spare.

**Validation against the brute force.** I ran the residue-bucket solution against the `O(n^2)` brute force on 500 random small cases (`n` up to ~20, `m` up to 15, `t` a valid target, frequencies up to ~100), deliberately keeping `m` small so self-pair residues and multiply-populated buckets occur often, and got zero mismatches. The small-`m` regime is exactly where the double-count and self-pair logic is stressed, so agreement there is strong evidence the case split is right; the separately-constructed `n = 2*10^5` all-even case validates the 64-bit arithmetic that the small random cases are too small to exercise.

**Final solution.** I convinced myself the *idea* is right by deriving the case split and reproducing the sample, I convinced myself the *counting code* is right by tracing the all-equal-residue input that broke the naive product loop and re-verifying the fix, and I convinced myself the *types* are right by constructing the `~2*10^10` worst case, watching the `int` version print garbage, and confirming the `long long` version prints the true value. That is what I ship — one self-contained file, the `O(n + m)` residue bucketing with every counting quantity in 64-bit:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m, t;
    if (!(cin >> n >> m >> t)) return 0;     // n machines, modulus m, target residue t

    // cnt[r] = how many frequencies are congruent to r modulo m.
    vector<long long> cnt(m, 0);
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        long long r = ((x % m) + m) % m;     // safe modulo for any sign (here x >= 0)
        cnt[r]++;
    }

    // We want pairs (i<j) with (a_i + a_j) % m == t.
    // Pair residues (r, s) with (r + s) % m == t. For r != s the count is cnt[r]*cnt[s]
    // (each unordered residue pair counted once); for the self-paired residue r == s it
    // is cnt[r]*(cnt[r]-1)/2 (choose 2 within the bucket).
    long long answer = 0;
    for (long long r = 0; r < m; r++) {
        long long s = ((t - r) % m + m) % m; // residue that completes r to t (mod m)
        if (r < s) {
            answer += cnt[r] * cnt[s];       // cnt[r]*cnt[s] can exceed 32-bit: long long
        } else if (r == s) {
            answer += cnt[r] * (cnt[r] - 1) / 2;
        }
        // r > s already handled when the loop variable was s
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The `O(n^2)` brute force is correct but `~2*10^10` operations too slow, so I bucket by residue mod `m`: a pair is resonant iff its residues `(r, s)` satisfy `r + s ≡ t`, reducing the count to a single pass over residues. My first product loop `answer += cnt[r]*cnt[s]` both double-counted off-diagonal residue pairs and treated the self-pair residue as `cnt[r]^2` instead of `C(cnt[r], 2)` — a trace of `[6,12,18]` returning `9` instead of `3` pinned it, fixed by the `r < s` / `r == s` / skip-`r > s` split. Then the constraints' real teeth: the answer reaches `C(2*10^5, 2) ≈ 2*10^10` and the per-bucket product `~4*10^10`, both past `INT_MAX`, so an `int` (which I confirmed prints `672547168` instead of `19999900000`) silently fails — every count, product, and accumulator is `long long`, and the completing residue is guarded with `((t - r) % m + m) % m` to keep the index non-negative; that closes the empty, single, `m = 1`, self-pair, max-frequency, and overflow corners.
