The input is a single integer `n` as large as `10^12`, and I have one second to print `D(n) = d(1) + d(2) + ... + d(n)`, the count of divisors summed over `1..n`. Two magnitudes fix the shape of any solution before I pick an algorithm. First, the size of the answer: `d(k)` averages about `ln k`, so `D(n) ≈ n ln n`, and at `n = 10^12` that is roughly `10^12 * 27.6 ≈ 2.8 * 10^13` — far past the `2.1 * 10^9` ceiling of 32-bit, so every accumulator has to be `long long`; an `int` anywhere on this path is a silent wrong answer on the largest test. Second, `n = 10^12` itself: any method whose running time is proportional to `n` is dead on arrival, so the real problem is finding something genuinely sublinear.

The textbook reading `D(n) = sum_k d(k)` invites a divisor sieve — for every `j`, walk its multiples and increment a count. That is `O(n log n)` time and `O(n)` memory, entirely correct, but at `n = 10^12` the array alone is `8` terabytes and the increments number `~4 * 10^13`: dead on both memory and time.

Counting divisor-incidences by the *divisor* rather than the dividend removes the array entirely. A fixed `i` divides exactly `floor(n/i)` of the integers in `1..n`, so

```
D(n) = sum_{i=1}^{n} floor(n / i).
```

This is one loop, `O(1)` memory — the memory problem is gone. But it is still `O(n)`: a trillion iterations, about `1000` seconds even at `10^9` iterations per second. The linear barrier has to be broken, not shaved with constants.

What lets me break it is the *structure* of the term sequence. Write out `floor(n/i)` for a small `n`, say `n = 30`:

```
i      : 1  2  3  4  5  6  7  8  9 10 11 ... 15 16 ... 30
floor  :30 15 10  7  6  5  4  3  3  3  2 ... 2  1 ...  1
```

The large values each occur at a single small `i`, while the small values repeat over long runs (`3` for `i` in `8..10`, `2` for `11..15`, `1` for `16..30`). This is general: `floor(n/i)` is a step function taking only about `2 * sqrt(n)` distinct values. So the trillion-term sum at `n = 10^12` really carries only `~2 * 10^6` distinct values, each over a block of consecutive `i` — and if each block costs `O(1)` the whole thing is `O(sqrt(n))`, a couple million operations.

I could chase down each constant run explicitly, but the geometry hands me something cleaner and symmetric: the Dirichlet hyperbola method. `D(n)` is the number of lattice points `(a, b)` with `a, b >= 1` and `a * b <= n` — each `k <= n` contributes one point per factorization `k = a*b`, i.e. `d(k)` points. That region is symmetric across `a = b`. Let `s = floor(sqrt(n))`. The vertical strip `a <= s` under the hyperbola holds `T = sum_{a=1}^{s} floor(n/a)` points (for each `a`, the partners are `b = 1 .. floor(n/a)`). By the `a <-> b` symmetry the horizontal strip `b <= s` holds the same `T`. Their union is the entire region; their overlap is exactly the points with both `a <= s` and `b <= s`, and since `a, b <= s` forces `a*b <= s*s <= n`, that overlap is the full `s × s` square. Inclusion–exclusion gives

```
D(n) = 2 * T - s*s,    T = sum_{i=1}^{s} floor(n / i),    s = floor(sqrt(n)).
```

That is `O(sqrt(n))` time, `O(1)` memory, and no array. Against `n = 6`: `s = 2`, `T = floor(6/1) + floor(6/2) = 9`, so `D = 18 - 4 = 14`, matching `d(1..6) = 1,2,2,3,2,4`.

The loop is trivial; the one line that can be silently wrong is `s = floor(sqrt(n))`. Floating `sqrtl((long double)n)` near `10^12` is only correct to within rounding, so it can land just above or just below the true root. If it rounds up, `s*s > n` and the `s × s` square pokes outside the hyperbola — the whole correction term is corrupt — and no small test would catch it, because the rounding only misbehaves at large `n`. So I do not trust the float; I clamp it to the exact integer floor with two adjustment loops that each run at most a step or two:

```
long long s = (long long)sqrtl((long double)n);
while (s * s > n) s--;                 // pull s down if sqrt rounded up
while ((s + 1) * (s + 1) <= n) s++;    // push s up if sqrt rounded down
```

After these, `s*s <= n < (s+1)*(s+1)` holds regardless of how the float rounded, and the guard products stay safe: the largest `s` is about `10^6`, so `(s+1)*(s+1) ≈ 10^12`, well inside `long long`. The correction is sharpest right at a perfect square, so I check the straddlers against the sieve. `n = 48`: `s = 6`, `T = 48+24+16+12+9+8 = 117`, `answer = 234 - 36 = 198`. `n = 49`: `s = 7`, `T = 125`, `answer = 250 - 49 = 201`. `n = 50`: `s = 7`, `T = 128`, `answer = 256 - 49 = 207`. The sieve gives `198, 201, 207` — agreement, because the clamp pins the `s × s` overlap to exactly the square under the hyperbola. At `n = 10^12` the clamp gives `s = 10^6` and the program returns `27785452449086`, matching an exact-`isqrt` Python evaluation of the same identity.

The remaining cases are quick. `n = 0` is an empty sum, so I special-case `n <= 0` and print `0` before ever calling `sqrt` (feeding `0` through the loop would also give `0`, but the early return documents the convention). `n = 1`: `s = 1`, `T = 1`, `answer = 2 - 1 = 1 = d(1)`. `n = 2`: `s = 1`, `T = 2`, `answer = 4 - 1 = 3 = d(1)+d(2)`. Empty input: `cin >> n` fails and I return without printing. On overflow, `2*T ≈ 5.7*10^13` is the largest quantity computed, four orders of magnitude below the `long long` limit.

To close it out I compiled with `g++ -O2 -std=c++17` and ran it against the `O(n log n)` divisor sieve over the explicit edges (small `n`, perfect squares, square-straddlers, powers-of-ten neighbours, primes like `99991`) plus a thousand random `n` up to `2*10^4`, the ceiling where the sieve stays fast: zero mismatches across `1028` cases. Because the sieve computes `D` by a completely different route, that agreement is real evidence rather than two copies of one bug. The `n = 10^12` run finishes in well under a millisecond on a few megabytes, so time and memory are met with enormous margin.

What I ship is exactly that, one self-contained file: the `n <= 0` early return for the empty-sum convention, the float `sqrt` clamped down to the exact integer floor `s`, the `O(sqrt(n))` loop over `i = 1..s` accumulating `n/i`, and the output `2*sum - s*s` — every accumulator a `long long`.
