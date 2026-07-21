The property I have to count is unusual in exactly one place: an integer `x` in `[L, R]` qualifies when its decimal digit sum `S(x)` is divisible by `len(x)`, its number of digits — and that divisor is not a fixed constant, it is a feature of the very number being tested. `19` passes because `2 | 10`, but `1900000000000000000` fails because `19` does not divide `10`, even though the two share the same multiset of digits. That the modulus is `len(x)`, decided by the whole number, is what lifts this above a stock digit-counting exercise, and it is where I expect all the difficulty to concentrate.

Before any logic, scale and types, because they decide everything downstream. `R` reaches `10^18`, which already needs 64 bits just to hold. Worse, the answer is a *count* of qualifiers up to `10^18`: roughly one number in `len` qualifies out of ~`10^18` numbers, so the count lands near `5·10^16`, an order of magnitude past the ~`2.1·10^9` ceiling of 32-bit. Every accumulator and the output must be `long long`; an `int` anywhere in the counting path is a silent wrong answer that passes every tiny test and loses only the large ones. I settle that before writing a line.

Counting over an arbitrary interval `[L, R]` is awkward; counting a prefix `[1, N]` is clean, because then I only ever reason about "numbers of at most this many digits, bounded above by `N`". So I take the standard difference: if `f(N)` counts balanced integers in `[1, N]`, the answer is `f(R) - f(L-1)`. The one care point is the bottom — when `L = 1`, `L - 1 = 0` — so I define `f(N) = 0` for `N <= 0` and the subtraction is always valid. Now the whole problem is one function of a single bound.

The most obvious `f(N)` is a loop over `x` from `1` to `N` testing `digitSum(x) % numDigits(x) == 0`. It is three lines and obviously correct — which is why I keep it as a brute oracle. As the solution it is hopeless: `N` is `10^18`, and even at an optimistic `10^9` operations per second a scan to `10^18` is over thirty years. No constant factor rescues a linear-in-`N` scan at this scale; I need cost that depends on the *number of digits* of `N` (about 19), not on `N`.

The textbook tool for "count integers `<= N` with a digit-defined property" is digit DP: write `N` in decimal, build candidates digit by digit from the most significant end, carrying a compact state `(position, running residue, tight)`, where `tight` marks whether the prefix so far equals `N`'s prefix (restricting the next digit) or has dropped strictly below (freeing it to `0..9`). For the classic "count `x <= N` with digit sum divisible by a *fixed* `m`", this is immediate: carry `S mod m`, branch over digits, count residue-`0` states at the end, all in `O(digits · m · 2)`, independent of `N`. That is exactly the asymptotics I want.

But my modulus is `len(x)`, and a most-significant-first walk does not know the final length while it is still choosing leading digits. The residue I should track is `S mod 5` if the number turns out to have five digits but `S mod 6` if it has six — different arithmetic, and I cannot commit to one because the length is not yet determined. If I pick a single modulus and run the DP, I count against the wrong divisor for numbers of every length but one. This is the non-locality biting: which divisor to use depends on a global property the per-digit walk hasn't fixed. Plain digit DP here is not slow, it is *wrong*.

The resolution: make the length known. Instead of counting all lengths at once, partition `[1, N]` by digit count — numbers with exactly `1`, `2`, …, up to `D = len(N)` digits. Within the part of exactly-`len`-digit numbers the modulus is a *constant*, namely `len`, and a number has exactly `len` digits precisely when its leading digit is nonzero across `len` positions. So each part is the well-understood classic problem: count `len`-digit numbers (leading digit `1..9`, the rest `0..9`), bounded above appropriately, whose digit sum is divisible by the fixed `len`. Sum over `len = 1..D`.

The bounds per part fall out cleanly. For any `len` strictly below `D`, *every* `len`-digit number is `< N` because `N` has more digits, so the bound is the maximal `len`-digit number `99…9` — effectively unrestricted. For the top length `len = D`, the `D`-digit numbers `<= N` are bounded by `N`'s own digits, so I run that one DP against `N`. Since each integer has exactly one length, the parts are disjoint and their union is all of `[1, N]` — no overlaps, no gaps. And the cost is negligible: at most `D <= 19` lengths, each DP `O(len · len · 10)`, well under `~70000` operations for the whole of `f(N)`, so two prefix evaluations per query are microseconds — comfortably inside the `10^18` / 1-second budget.

For each fixed `len` with `mod = len` I need: among `len`-digit numbers (no leading zero) that are `<= bound`, how many have digit sum `≡ 0 (mod mod)`? I use the standard "walk the tight prefix, and whenever a digit drops strictly below the bound digit, multiply in the free completions" structure. To make the completions O(1), I precompute a suffix table `suf[p][s]` = the number of ways to fill positions `p..len-1` with arbitrary digits `0..9` so that *that suffix alone* has digit sum `≡ s (mod mod)`. Then I walk positions left to right keeping the tight prefix's running residue `r`. At position `p` with bound digit `bd[p]`, every digit `d` strictly less than `bd[p]` (starting at `1` when `p = 0`, no leading zero) frees the suffix from `p+1` on and contributes `suf[p+1][need]`, where `need = (mod - (r+d) mod mod) mod mod` is the suffix residue that lands the total on `0`. After the strictly-less digits, I continue the tight path with `d = bd[p]`, updating `r`. At the end, if the tight path stayed feasible and its residue is `0`, the bound itself qualifies and I add `1`. The only ingredient specific to this problem is that `mod` equals `len`.

The suffix recurrence is the part I have to get exactly right. Base case: `suf[len][0] = 1`, all other `suf[len][s] = 0`, since the empty suffix has digit sum `0`. For the step, placing digit `d` at position `p` leaves the rest needing digit sum `≡ s - d`, so

```
suf[p][s] = sum over d in 0..9 of suf[p+1][(s - d) mod mod]
```

with the index reduced into `[0, mod)`. In C++ `%` of a negative left operand is negative, so it has to be written `((s - d) % mod + mod) % mod`. The direction of that subtraction is the one place the whole DP can quietly go wrong: `s - d` is what my `need` lookup will assume the table means, but `s + d` reads just as plausibly at the keyboard.

I write `countLen(len, mod, bd)` building `suf` and walking the tight path, `countUpTo(N)` looping `len = 1..D` with the all-nines bound for `len < D` and `N`'s digits at `len = D`, and `main` doing `f(R) - f(L-1)`. It compiles clean under `-O2 -std=c++17`. Now the check that matters: differential testing against the scan oracle. Hand cases pass — `1 20 -> 15`, `1 1 -> 1`, `9 9 -> 1`, `10 10 -> 0`. Then 600 random small ranges, and it breaks immediately:

```
MISMATCH seed=4 input=[997 1012] sol=5 brute=4
MISMATCH seed=5 input=[133927 231154] sol=16203 brute=16195
...
```

The solution overcounts. The smallest failure `[997, 1012]` gives `sol=5`, `brute=4`. The four the brute marks balanced are `999` (sum `27`, len `3`), `1003`, `1007`, `1012` (each sum divisible by 4). To find the phantom fifth I evaluate the solution point by point across the window — `f(x) - f(x-1)` is `1` exactly when `x` is balanced — and it flags `1010`, whose digit sum is `2`, length `4`, `2 % 4 = 2 ≠ 0`. So `1010` is not balanced and the bug lives in the `len = 4` DP near the bound `1010`.

I compare `N = 1010` against `N = 1009`: for `len = 4`, `1009` contributes `2` correctly (`1003`, `1007`), but `1010` contributes `3`. So `countLen(4, 4, [1,0,1,0])` returns `3` where the truth is `2`. Tracing it:

```
p=0 tight d=1 -> r=1
p=1 tight d=0 -> r=1
p=2 place d=0 (< bd=1): need suf[3][3] = 3   <-- here
p=2 tight d=1 -> r=2
p=3 tight d=0 -> r=2
total = 3
```

The suspicious term is `suf[3][3] = 3`. Position `3` is the last digit, so `suf[3][need]` should count single digits whose own value is `≡ need (mod 4)`; for `need = 3` that is `{3, 7}`, exactly `2`, not `3`. The table is returning the count for the wrong residue. Looking at how I built it in the first cut:

```
for d in 0..9: acc += suf[p+1][(s + d) % mod];   // wrong sign
```

I used `(s + d)`. But under my intended meaning — `suf[p][s]` counts suffixes whose own sum is `≡ s` — placing `d` leaves residue `s - d` for the rest, so the index must be `(s - d) mod mod`. The `+d` form silently builds a *different* table, the one for residue `-s`; by the symmetry of the digit set `{0..9}` the two agree for most residues, which is exactly why the hand cases and most random cases passed and hid the bug. For `mod = 4`, `s = 3` they diverge: the buggy table reports `3` (digits `{1,5,9}`, residue `1 ≡ -3`) where the truth is `2` (digits `{3,7}`). That is the phantom `1010`. The lookup (`need`, asking for suffix-sum `≡ need`) and the build disagreed on what the index means, and it was the build that had the sign flipped.

I change the build to `((s - d) % mod + mod) % mod` and recompile. The previously failing cases clear: `[997,1012] -> 4`, `[1010,1010] -> 0`, `[1,1010] -> 356` matching brute; the full sweep of 600 random small ranges returns zero mismatches.

The scan oracle can only reach small ranges, but the real constraint is `10^18`, where lengths run to 19 and moduli to 19 — territory the small sweep never touches. So I write a second, independent checker in a different style: a recursive memoized per-length digit DP `rec(pos, residue, tight)` accumulating `(res + d) % mod` forward. Both agree on `[1, 10^18] -> 55919522262267858`; spot ranges `[123456789, 987654321098765432]`, `[5·10^17, 10^18]`, the single point `[10^18-1, 10^18]`, and 300 random full-width `[L, R]` pairs all match with zero mismatches. Two implementations built from different directions agreeing on the extreme range is the evidence I trust for both the length decomposition and the per-length DP.

The edge cases land where the decomposition was designed to make them land. `L = 1` gives `countUpTo(0) = 0` by the guard, so `[1,1] -> 1`. Single-digit numbers have `len = 1`, modulus `1`, everything divisible, so `[1,9] -> 9`. A power-of-ten boundary, where the divisor changes mid-range, is the whole reason the length split exists: `[999, 1000] -> 1`, only `999` qualifying since `1000` has sum `1` over length `4`. At the extreme `R = 10^18` the 19-digit tight path handles the top length and the count `55919522262267858` sits an order of magnitude below the `~9.2·10^18` signed-64-bit ceiling, so no accumulator overflows, and the run finishes under a millisecond.

So the shape that survives is the length decomposition wrapping a per-length suffix-table digit DP, with the `s - d` recurrence sign the one thing I had to catch by testing; the full module is in the answer.
