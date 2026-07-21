The string `s` has length `n` with `n <= 5000`, and I must pack non-overlapping *squares* —
substrings of the form `uu`, even length `>= 2`, first half equal to second — so that the total
number of covered characters is as large as possible. Coverage can never exceed `n = 5000`, so the
answer itself sits in an `int`; what the bound `n <= 5000` actually pins down is the time budget. An
`O(n^2)` sweep is about `2.5*10^7` steps, comfortable inside 2 seconds — provided each step is
`O(1)`. If each square test instead compares its two halves character by character, the cost climbs
to `O(n^3) ≈ 10^{11}` and the whole thing is hopeless. So the problem collapses to one requirement: a
constant-time substring-equality test. That is exactly what a polynomial rolling hash provides, and
it is the reason the hash is on the table at all. The empty selection covers `0`, so the answer is
never negative; I read `n`, read the string only when `n > 0`, and print a single integer.

Two routes could solve this, and I want the one I can prove correct rather than the one that is
shortest to type.

- *Left-to-right greedy.* Scan from the left; at the first position where a square begins, grab one
  (say the shortest, to leave the most room), jump past it, repeat. Near-linear, a dozen lines. The
  worry is structural: squares interlock. A short square starting at position `p` overlaps a longer
  square starting at `p+1`, and taking the short one early forbids the long one. Greedy decides
  locally; coverage is global. That is precisely the setting where a local grab can cost more than it
  gains, so I will not trust greedy until I have tried to break it.
- *Prefix DP.* Let `dp[i]` be the best coverage using non-overlapping squares lying entirely inside
  `s[0..i)`. To reach `i`, either leave position `i-1` uncovered (`dp[i-1]`) or close a square ending
  exactly at `i`: for each `L >= 1` with `2L <= i` where the two halves `s[i-2L..i-L)` and
  `s[i-L..i)` match, `dp[i]` can be `dp[i-2L] + 2L`. That is `O(n)` candidate lengths per `i`,
  `O(n^2)` total, each tested in `O(1)` by hashing.

So I try to break greedy first. Take `s = "aabab"`, indices `0..4`. The squares: `aa` at `[0,2)`
(halves `a`,`a`) and `abab` at `[1,5)` (halves `ab`,`ab`); `s[2..4)="ba"` and `s[3..5)="ab"` are not
squares, and odd-length windows cannot be. So exactly two squares, and they overlap at position `1`.
Greedy-shortest finds the square beginning at index `0`, grabs `aa`, covers `2`, and jumps to index
`2`; from there the remaining `bab` has no square starting in it, so greedy halts at `2`. But
skipping `aa` and taking `abab` covers `4`. Greedy-longest is no better — at index `0` the only
square is `aa`, so it grabs the same piece and lands on the same `2`. The interlock I worried about
is real: a single early grab at position `0` blocked position `1` and forbade the longer square that
started one step later. Greedy is out; I commit to the DP.

Now the DP itself. The only thing the future needs to know about a chosen layout inside `s[0..i)` is
its total coverage — there is no "last square" to carry, because a square closing at `i` occupies
`[i-2L, i)` and everything before it is accounted for by `dp[i-2L]`, which by construction only
touches positions `< i-2L`, so non-overlap is automatic. The state is just the prefix length `i`.
Transitions: `dp[i] >= dp[i-1]` (any layout of `s[0..i-1)` is a layout of `s[0..i)` with `i-1` left
bare), and `dp[i] >= dp[i-2L] + 2L` whenever `s[i-2L..i)` is a square. Base `dp[0] = 0`, answer
`dp[n]`. On `aabab` this bottoms out right: the length-2 square at the front lifts `dp[2]` to `2`,
that value stalls through `dp[3]` and `dp[4]`, and at `dp[5]` the halves `s[1..3)="ab"` and
`s[3..5)="ab"` match, giving `dp[1] + 4 = 4` — the coverage that beat greedy.

The DP needs `sub(l,r)` = hash of `s[l..r)` in `O(1)`. I use prefix hashes `h[i]` = hash of
`s[0..i)` and powers `pw[k] = base^k`, all modulo the Mersenne prime `2^61-1` so reduction is a
shift-and-add: `h[i+1] = h[i]*base + (s[i]+1)`, and `sub(l,r) = h[r] - h[l]*base^{r-l}`. The trap
here is not the formula but the arithmetic. Every `h` is reduced into `[0, MOD)`, so in the
subtraction `h[r] - h[l]*pw[r-l]` the minuend can be the *smaller* term; computed in unsigned 64-bit
that wraps around `2^64`, and since the modulus is `2^61-1` the wrapped value is not congruent to the
true residue — two equal substrings would then hash unequal (or worse), and the DP would silently
miscount. The fix is to bias positive before subtracting: with `x = mulmod(h[l], pw[r-l]) < MOD`,
form `res = h[r] + MOD - x`, which lies in `[1, 2*MOD)`, then subtract `MOD` once if needed. The
`mulmod` for this modulus multiplies into a `__uint128_t` and reduces by `(c & MOD) + (c >> 61)` —
congruent because `2^61 ≡ 1 (mod 2^61-1)` — with one conditional subtraction to finish.

The square test has one index that is easy to slip on: a square of length `2L` ending at `i` starts
at `j = i - 2L`, with halves `s[j..j+L)` and `s[j+L..i)`, *both* of length `L`. The second half must
end at `i = j+2L`, not one past it; drift that end index by one and the two "halves" have different
lengths, every length-2 square goes undetected, and the answer collapses — an all-`a` string reports
`0` instead of `n`. So the inner loop compares `sub(j, j+L)` against `sub(j+L, i)` and, on a match,
offers `dp[j] + 2L` as a candidate for `dp[i]`. Run on the two smallest revealing cases it gives `4`
on `aabab` (the interlock) and `4` on `aaaa` (two `aa`s tiled, the length-2 detection).

A few edge cases remain to pin down. `n = 0`: I read `n`, skip the string
entirely (no token follows), neither the hash loop nor the DP loop runs, and I print `dp[0] = 0`. To
keep input handling from reading past `s` on a stray length mismatch I trust the actual string and
reset `n = s.size()` after reading. `n = 1` (`s = "a"`): no even window exists, `dp[1] = dp[0] = 0` —
a lone character is not a square. A square-free string like `abc` keeps `dp` at `0` throughout. An
all-equal string tiles completely (`aaaaaa -> 6`), which I checked against a brute force. The worst
case for the inner loop is `n = 5000` all `a`, where every `L` is a square and no structure prunes
the sweep: about `1.25*10^7` `sub` calls, which runs in roughly `0.02 s` and a few MB — well inside
both limits.

Correctness finally rides on the hash. Equal substrings always hash equal — the formula is exact —
so I never miss a real square; the only risk is a false positive, two unequal halves colliding, at
probability about `1/MOD ≈ 4*10^{-19}` per comparison. Across the `~1.25*10^7` comparisons at
`n = 5000` the union bound gives `~5*10^{-12}`, negligible, so a single 61-bit modulus suffices
rather than a double hash. Mapping each character to `s[i]+1 >= 1` stops a run of "zero"-valued
leading characters from collapsing strings of different length, and drawing the base from a fixed
`mt19937_64` seed keeps the program deterministic while denying an adversary a known base to
hand-craft a collision against.
