**Reading the problem and pinning the contract.** I am given a range `[L, R]` with `1 <= L <= R <= 10^18`, and I must count the integers `x` in that range that are *balanced*: their decimal digit sum `S(x)` is divisible by their digit count `len(x)`, i.e. `S(x) mod len(x) == 0`. Input is two integers on stdin, output is one integer. The very first thing I fix is scale and types, because they decide everything downstream. `R` is up to `10^18`, which already needs 64-bit just to hold. And the *answer* is a count of qualifying integers up to `10^18`; even a rough estimate — roughly one number in `len` qualifies, and there are `~10^18` numbers — puts the count near `5 * 10^16`, far past the `~2.1 * 10^9` ceiling of 32-bit. So the output and every accumulator must be `long long`. An `int` anywhere in the counting path is a silent wrong-answer on the large tests. That decision is non-negotiable and I make it before writing any logic.

**Reducing the range to a prefix count.** Counting over an arbitrary interval `[L, R]` is awkward; counting over a prefix `[1, N]` is clean because then I only ever reason about "numbers with at most this many digits, bounded above by `N`". The standard move is inclusion by subtraction: if `f(N)` counts balanced integers in `[1, N]`, then the answer for `[L, R]` is `f(R) - f(L-1)`. I have to be a little careful at the bottom: when `L = 1`, `L - 1 = 0`, and `f(0)` must be `0` (there are no positive integers `<= 0`). I will define `f(N) = 0` for `N <= 0` so the subtraction is always valid. Good — the whole problem now reduces to computing `f(N)` for a single bound, and I can think purely about prefixes.

**The obvious approach, and why it dies.** The most obvious `f(N)` is a loop: for `x` from `1` to `N`, compute the digit sum and digit count and test divisibility. It is three lines and obviously correct — in fact I will use exactly this as my brute-force oracle later. But as the actual solution it is hopeless. `N` can be `10^18`. Even at an optimistic `10^9` simple operations per second, scanning to `10^18` takes about `10^9` seconds, which is over thirty years. There is no constant factor that rescues a linear-in-`N` scan when `N` is `10^18`. I need something whose cost depends on the *number of digits* of `N` (about 19), not on `N` itself.

**Reaching for digit DP, and hitting the real obstacle.** The textbook tool for "count integers `<= N` with a digit-defined property" is digit dynamic programming. The idea: write `N` in decimal, then build candidate numbers digit by digit from the most significant end, carrying a compact state. The usual state is `(position, runningValue mod m, tight)`, where `tight` says whether the prefix built so far is exactly equal to `N`'s prefix (which restricts the next digit) or already strictly below it (which frees the next digit to be anything `0..9`). For the classic problem "count `x <= N` with digit sum divisible by a *fixed* modulus `m`", this is immediate: carry `S mod m`, branch over digits, and at the end count the states with residue `0`. The cost is `O(digits * m * 2)`, completely independent of `N`'s magnitude. That is exactly the asymptotics I need.

But my problem is not the classic one, and the difference is the whole point. My modulus is `len(x)` — the number of digits of the candidate — and digit DP, walking from the most significant digit, *does not know the final length while it is still choosing leading digits*. Concretely, consider a number with a leading run that could be a `5`-digit number or, with different choices, conceptually part of a longer string. The residue I should be tracking is `S mod 5` if the number turns out to have `5` digits but `S mod 6` if it has `6` digits — and these are different arithmetic. I cannot carry "digit sum mod the length" as a single residue, because the length is not yet determined. If I naively pick one modulus and run the DP, I count against the wrong divisor for numbers of every length except one. This is the non-locality biting: the rule (which divisor) depends on a global property (the length) that the per-digit walk hasn't committed to. So plain digit DP, applied directly, is simply *wrong* here, not merely slow.

**Deriving the insight: fix the length, and the modulus stops moving.** Here is the resolution, and it is the kind of move that feels obvious only once you see it. The obstacle is that the modulus is unknown because the length is unknown. So I *make the length known* — I stop trying to count all lengths at once. I partition `[1, N]` by digit count: numbers with exactly `1` digit, exactly `2` digits, …, up to exactly `D` digits where `D = len(N)`. Within one part — say, the part of all numbers that have exactly `len` digits — the modulus is a *constant*: it is `len`. And a number has "exactly `len` digits" precisely when its leading digit is nonzero and it has `len` positions. So inside each part I am back to the well-understood classic problem: count `len`-digit numbers (leading digit `1..9`, the rest `0..9`), bounded above by the appropriate limit, whose digit sum is divisible by the *fixed* modulus `len`. I solve that with a clean per-length digit DP, then sum the answers over `len = 1 .. D`.

This decomposition is exact and the bounds per part are easy. For a length `len` strictly less than `D = len(N)`: *every* `len`-digit number is `<= N`, because `N` has more digits and is therefore larger than any `len`-digit number. So the upper bound for that part is the largest `len`-digit number, `99...9` (`len` nines) — effectively unrestricted, every `len`-digit number is in range. For the top length `len = D`: the numbers with `D` digits that are `<= N` are bounded by `N` itself, so I run the digit DP with `N`'s own digits as the bound. That single special case at the top, plus a free (all-nines bound) DP for each shorter length, covers `[1, N]` exactly with no overlaps and no gaps — each integer has exactly one length, so the parts are disjoint and their union is everything.

The cost is tiny. There are at most `D <= 19` lengths. For each, the digit DP is `O(len * len * 10)` (positions × residues × digit choices), so well under `19 * 19 * 10` per length and under `~70000` operations total for the whole of `f(N)`. With two prefix evaluations per query this is microseconds. The asymptotics comfortably clear the `10^18` constraint, which is the thing to obsess over: this is the standard, strongest formulation for a length-dependent digit property, and it has exactly the right complexity to pass.

**Choosing the digit-DP formulation: tight-prefix walk plus free-suffix counts.** For each fixed length `len` with modulus `mod = len`, I need: among `len`-digit numbers (no leading zero) that are `<= bound`, how many have digit sum `≡ 0 (mod mod)`? I implement this with the standard "walk the tight prefix, and whenever a digit goes strictly below the bound digit, multiply in the number of free completions" structure. To make the "free completions" part O(1) to look up, I precompute a suffix table:

- `suf[p][s]` = the number of ways to fill positions `p .. len-1` with arbitrary digits `0..9` such that the digit sum of *that suffix alone* is `≡ s (mod mod)`.

Then I walk positions left to right keeping the running residue `r` of the tight prefix. At position `p`, the bound digit is `bd[p]`; for every digit `d` strictly less than `bd[p]` (and respecting "no leading zero" at position `0`, where `d` starts at `1`), the prefix becomes free from `p+1` onward. Such a choice contributes `suf[p+1][need]` completions, where `need` is the suffix residue that makes the total digit sum land on `0`: if the prefix-plus-`d` residue is `nr = (r + d) mod mod`, then the suffix must contribute `need = (mod - nr) mod mod`. After exhausting the strictly-less digits at position `p`, I continue the tight path by taking `d = bd[p]` itself, updating `r`. At the very end, if the whole tight path stayed feasible and its total residue is `0`, the bound number itself qualifies and I add `1`. This is the canonical, clean digit DP; the only ingredient specific to this problem is that `mod` equals `len`.

**Building `suf` — the recurrence I have to get exactly right.** The base case is `suf[len][0] = 1` and `suf[len][s] = 0` for `s != 0`: the empty suffix has digit sum `0`. For the inductive step, I want `suf[p][s]` = number of suffix strings over positions `p..len-1` whose own digit sum is `≡ s`. If I place digit `d` at position `p`, then the rest (positions `p+1..len-1`) must carry digit sum `≡ s - d`, so:

```
suf[p][s] = sum over d in 0..9 of suf[p+1][(s - d) mod mod]
```

with the subtraction taken modulo `mod` into `[0, mod)`. I write that out carefully as `((s - d) % mod + mod) % mod` to keep the index non-negative in C++ (where `%` of a negative left operand is negative). This recurrence and this index are the part I expect to fumble, so I will test it hard.

**First implementation.** I write `countLen(len, mod, bd)` building `suf` and walking the tight path, and `countUpTo(N)` looping `len = 1..D` with the all-nines bound for `len < D` and `N`'s digits for `len = D`, summing the per-length counts. Then `main` does `f(R) - f(L-1)`. I compile with `-O2 -std=c++17` and it builds clean. Now the part that matters: differential testing against the slow-but-obvious oracle.

**The brute-force oracle and the first stress run.** My oracle is the doomed-as-a-solution-but-perfect-as-a-check loop: for `x` in `[L, R]`, test `digitSum(x) % numDigits(x) == 0`. I keep generated ranges small enough to scan, then compare. The hand cases pass — `1 20 -> 15`, `1 1 -> 1`, `9 9 -> 1` (digit sum `9`, length `1`, divisible), `10 10 -> 0` (sum `1`, length `2`). Encouraging. Then I run 600 random small ranges, and it immediately breaks:

```
MISMATCH seed=4 input=[997 1012] sol=5 brute=4
MISMATCH seed=5 input=[133927 231154] sol=16203 brute=16195
...
```

So my solution overcounts. Good — the oracle earned its keep on the very first batch.

**Debugging: localize, then trace.** The smallest failure is `[997, 1012]`: `sol=5`, `brute=4`. I list which numbers the brute marks balanced in that window: `999` (sum `27`, len `3`, `27 % 3 = 0`), `1003` (sum `4`, len `4`), `1007` (sum `8`, len `4`), `1012` (sum `4`, len `4`). That is `4`, correct. To find the phantom fifth, I evaluate my solution on each single point `x x` across the window (since `f(x) - f(x-1)` is `1` exactly when `x` is balanced). The single-point scan shows my solution flags `1010` as balanced — but `1010` has digit sum `1+0+1+0 = 2`, length `4`, and `2 % 4 = 2 != 0`. So `1010` is *not* balanced; my solution wrongly counts it. The bug lives in the `len = 4` digit DP near the bound `1010`.

**Finding the precise defect.** I instrument `countUpTo` to print the per-length contribution and compare `N = 1010` against `N = 1009`. For `len = 4`, `N = 1009` gives contribution `2` (correctly: `1003`, `1007`), but `N = 1010` gives `3` — one too many. So `countLen(4, 4, [1,0,1,0])` returns `3` when the truth (4-digit numbers `<= 1010` with digit sum divisible by 4) is exactly `2`. I add a line-by-line trace inside `countLen` for `bd = [1,0,1,0]`:

```
p=0 tight d=1 -> r=1
p=1 tight d=0 -> r=1
p=2 place d=0 (< bd=1): need suf[3][3] = 3   <-- here
p=2 tight d=1 -> r=2
p=3 tight d=0 -> r=2
total = 3
```

The single suspicious term is `suf[3][3] = 3`. Position `3` is the last digit; `suf[3][need]` is supposed to be "single digits whose own value is `≡ need (mod 4)`". For `need = 3` that is the digits `{3, 7}` — exactly `2` of them, not `3`. My `suf` table is returning the count for the *wrong* residue.

**Diagnosing the root cause.** I look at how I built `suf` in the first cut:

```
suf[npos][0] = 1;
for p from npos-1 down to 0:
  for s in 0..mod-1:
    acc = 0;
    for d in 0..9: acc += suf[p+1][(s + d) % mod];   // <-- wrong sign
    suf[p][s] = acc;
```

I used `(s + d) % mod`. But with my intended meaning — `suf[p][s]` = suffixes whose own digit sum `≡ s` — placing digit `d` at position `p` leaves residue `s - d` for the rest, so the recurrence must index `suf[p+1][(s - d) mod mod]`, not `(s + d)`. The `+d` version computes a *different* table: it makes `suf[p][s]` count suffixes whose digit sum `≡ -s (mod mod)` instead of `≡ s`. For most residues the two tables happen to agree by symmetry of the digit set, which is exactly why the small hand cases and many random cases passed and masked the bug — but for `mod = 4` and `s = 3` they differ: my buggy table reported `3` (the count for residue `-3 ≡ 1`, i.e. digits `{1,5,9}`) where the correct count for residue `3` is `2` (digits `{3,7}`). That mismatch is the phantom `1010`. So the lookup semantics (`need = (mod - nr) % mod`, asking for suffix-sum `≡ need`) and the table's build recurrence disagreed on the meaning of the index. One of them had the sign flipped, and it was the build.

**The fix and re-verification.** I change the build to match the lookup's meaning:

```
int ns = ((s - d) % mod + mod) % mod;
acc += suf[p + 1][ns];
```

Recompile, and re-run the previously failing cases: `[997,1012] -> 4` (was `5`), `[1010,1010] -> 0` (was `1`), `[1,1010] -> 356` matching brute. Then the full sweep: 600 random small ranges against the scan oracle — `TOTAL=600 MISMATCHES=0`. The bug is fixed, and it was fixed for the reason I identified (a sign error in the residue index of the suffix recurrence), which is the evidence I trust over "it passes now".

**Pushing past what the scan oracle can reach.** The scan brute can only check small ranges, but the real constraint is `10^18`, where the interesting behavior (lengths up to 19, modulus 19) lives. So I write a *second, independent* checker using a completely different style — a recursive memoized digit DP per length, `rec(pos, residue, tight)` accumulating `(res + d) % mod` forward — and cross-check my iterative solution against it on large inputs. The results line up exactly: `[1, 10^18] -> 55919522262267858` from both; spot ranges like `[123456789, 987654321098765432]`, `[5*10^17, 10^18]`, and the single-point `[10^18-1, 10^18]` all agree; 300 random full-width `[L, R]` pairs over `[1, 10^18]` give `MISMATCHES=0`. Two independent implementations agreeing on the extreme range is strong evidence the length decomposition and the per-length DP are both right.

**Edge cases, deliberately.**
- `L = 1`: `L - 1 = 0`, and `countUpTo(0) = 0` by the `N <= 0` guard, so the subtraction is well defined and the bottom of the range is included correctly (`[1,1] -> 1`).
- Single-digit numbers: `len = 1`, modulus `1`, and every integer is divisible by `1`, so all of `1..9` are balanced — confirmed (`[1,9] -> 9`).
- A power-of-ten boundary, where the length and therefore the divisor changes mid-range: `[999, 1000]` returns `1` (only `999` qualifies; `1000` has sum `1`, length `4`), matching brute. This is the case the whole length-decomposition exists to handle, and it is correct.
- The extreme bound `R = 10^18`: 19 digits, handled by the `len = D = 19` tight path; the count `~5.6 * 10^16` fits comfortably in `long long`, and there is no overflow because every accumulator is 64-bit and the counts never approach `9.2 * 10^18`.
- Performance: the worst case `1 1000000000000000000` runs in under a millisecond with a few megabytes of memory — the cost is `O(D^2 * 10)` per prefix, independent of `N`'s magnitude, exactly as designed.

**Final solution.** I convinced myself the *idea* is right by seeing why plain digit DP cannot carry a length-dependent modulus and by decomposing into fixed-length parts where the modulus is constant; I convinced myself the *code* is right by tracing a concrete overcount (`1010`) to a precise sign error in the suffix recurrence, fixing it, and then differentially testing against two independent oracles — a brute scan on small ranges and a separate recursive digit DP up to `10^18` — with zero mismatches over 900 cases. That is what I ship: one self-contained C++17 file.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count integers x in [1, N] (N >= 0) whose decimal digit sum is divisible by
// the number of decimal digits of x (its length, no leading zeros).
//
// The divisor depends on the length of x, which is non-local, so we cannot run
// one digit DP with a single modulus. Instead we group by length: for every
// target length len (1..19) the modulus is exactly len and fixed, so a standard
// digit DP over numbers of that length works. We count, for each len, how many
// numbers with EXACTLY len digits (leading digit 1..9) lie in [1, N] and have
// digit sum divisible by len.

typedef long long ll;

// Count numbers of exactly `len` digits, value <= bound, no leading zero,
// digit sum divisible by `mod` (== len). `bd` is the digit array of the upper
// bound (most significant first) of length exactly `len`. Returns the count.
//
// Technique: tight-prefix walk + precomputed free-suffix counts. We precompute
// suf[p][s] = number of ways to fill positions p..len-1 with free digits 0..9
// whose own digit sum is ≡ s (mod). Then we walk the tight prefix; at each
// position, every digit strictly below the bound digit "releases" a free suffix,
// whose count we read straight out of suf[]. Finally we add the bound itself if
// it qualifies.
ll countLen(int len, int mod, const vector<int>& bd) {
    int npos = len;
    // suf[p][s] = number of digit strings for positions p..len-1 (each digit
    // 0..9) whose OWN digit sum ≡ s (mod `mod`). Recurrence: placing digit d at
    // position p needs the remaining suffix (p+1..) to have digit sum ≡ s-d.
    vector<vector<ll>> suf(npos + 1, vector<ll>(mod, 0));
    suf[npos][0] = 1; // empty suffix has digit sum 0
    for (int p = npos - 1; p >= 0; --p) {
        for (int s = 0; s < mod; ++s) {
            ll acc = 0;
            for (int d = 0; d <= 9; ++d) {
                int ns = ((s - d) % mod + mod) % mod;
                acc += suf[p + 1][ns];
            }
            suf[p][s] = acc;
        }
    }
    // Walk the tight path. At position p, residue r (digit sum so far mod `mod`).
    // We must respect: at position 0 the digit range is 1..9 (no leading zero),
    // at positions >0 it is 0..9.
    ll total = 0;
    int r = 0;        // current residue along the tight prefix
    bool feasible = true; // whether the exact-bound path is still alive
    for (int p = 0; p < npos; ++p) {
        int lo = (p == 0) ? 1 : 0;
        int hi = bd[p];
        // place digit d in [lo, hi-1] strictly less than bound digit -> free suffix
        for (int d = lo; d < hi; ++d) {
            int nr = (r + d) % mod;
            int need = (mod - nr) % mod; // suffix residue needed to total 0
            total += suf[p + 1][need];
        }
        // continue tight with d == bd[p], but only if it respects lo
        if (bd[p] < lo) { feasible = false; break; }
        r = (r + bd[p]) % mod;
    }
    if (feasible && r == 0) total += 1; // the bound itself, if its digit sum ≡ 0
    return total;
}

// Count x in [1, N] with digitSum(x) % len(x) == 0.  N >= 0.
ll countUpTo(ll N) {
    if (N <= 0) return 0;
    // decompose N into digits
    string s = to_string(N);
    int D = (int)s.size(); // number of digits of N
    ll ans = 0;
    // For lengths 1 .. D-1: all numbers of that length are fully below N's range
    // upper bound is the maximal len-digit number (all 9s), i.e. unrestricted.
    // For length D: bounded by N.
    for (int len = 1; len <= D; ++len) {
        int mod = len; // length-dependent modulus
        vector<int> bd(len);
        if (len < D) {
            // upper bound = 99...9 (len nines): every len-digit number qualifies
            for (int i = 0; i < len; ++i) bd[i] = 9;
        } else {
            for (int i = 0; i < len; ++i) bd[i] = s[i] - '0';
        }
        ans += countLen(len, mod, bd);
    }
    return ans;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ll L, R;
    if (!(cin >> L >> R)) return 0;
    ll ans = countUpTo(R) - countUpTo(L - 1);
    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** A direct scan to `10^18` is thirty-plus years of compute, so it is out; plain digit DP is the right asymptotics but cannot carry a modulus that equals the candidate's own length, because the length is undecided while leading digits are still being chosen — so it is *wrong*, not just slow. The fix is to partition `[1, N]` by digit count: within a fixed length the modulus is the constant `len`, reducing each part to the classic "digit sum divisible by a fixed modulus" DP, with the all-nines bound for every length below `len(N)` and `N`'s own digits for the top length. Implementing that, a sign error in the suffix recurrence (`s + d` where it had to be `s - d`) overcounted `1010` and showed up first on `[997, 1012]`; tracing `suf[3][3]` to the wrong residue pinned it, the one-character fix cleared all 600 small cases against a brute scan, and an independent recursive digit DP confirmed agreement up to `10^18` over 300 more — `~5.6 * 10^16` qualifying integers in `[1, 10^18]`, all inside 64-bit.
