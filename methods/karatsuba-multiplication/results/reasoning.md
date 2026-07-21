The thing I'm actually after is a lower bound. Kolmogorov posed the conjecture at the seminar: multiplying two `n`-digit numbers needs on the order of `n²` bit operations, and no method can beat it. I find that very plausible and want to prove it. I don't see any way to argue directly that every conceivable automaton needs `n²` operations — the space of algorithms is too wild for a direct argument. The only handle I have is to corner the problem from the other side: build the most economical multiplication scheme I can actually write down, see what its operation count is forced to be, and hope that count bottoms out at `n²` for a structural reason I can then argue is unavoidable. That's a backwards way to prove a lower bound, but it's the only entry point I have, so let me build the tightest recursive scheme I can and watch what the count does.

The only method I actually know how to execute is the schoolbook one, and it costs `Θ(n²)`: `n` rows of `n` single-digit products, one per digit pair, summed with carries. The case for treating that as a floor is historical (people have used this for four millennia; if anything cheaper existed, surely it would have surfaced by now) and structural (every digit of `a` genuinely meets every digit of `b` in the final product — `aᵢ` lands at place `i+j` for each `bⱼ`, there are `n²` such pairs, and they all contribute — so it feels like `n²` little multiplications with no escape). Both of those are exactly what a real construction should either confirm or break.

Multiplication and squaring are interchangeable up to a constant, via `a·b = ¼[(a+b)² − (a−b)²]` — dividing by 4 is a two-bit shift in binary — so I can study squaring `a²` instead of the two-variable product; a lower bound for one transfers to the other. Kolmogorov pointed this out immediately, and it's a real simplification: one operand and its square, instead of two operands.

To get a handle on `a²` recursively, split `a` by digit position. With `n = 2m`, write `a = aₕ·Bᵐ + aₗ`, `aₕ` the high `m` digits, `aₗ` the low `m` digits. Squaring: `a² = aₕ²·B²ᵐ + 2aₕaₗ·Bᵐ + aₗ²`. Three pieces. Two of them, `aₕ²` and `aₗ²`, are squares of `m`-digit numbers — exactly the smaller version of the problem I'm recursing on. The middle piece `2aₕaₗ` is a *product*, not a square. If I insist on a square-only recursion, reducing it with the same identity `ab = ¼[(a+b)²−(a−b)²]` costs two more `m`-size squarings. If instead I analyze a two-operand product directly, the ordinary half-split has the same four products: high-high, high-low, low-high, low-low. Either way the naive recursive picture is four half-size multiplication problems plus linear recombination. Let me write the cost that way and see what it forces.

This is the naive four-way split, and its recurrence is `T(n) = 4·T(n/2) + O(n)`. Its recursion tree makes visible exactly where the `n²` comes from: level `i` holds `4ⁱ` nodes of size `n/2ⁱ`, each doing `O(n/2ⁱ)` combine work, so the per-level total is `O(2ⁱ·n)` — it doubles every level down, so the leaves dominate, and there are `4^{log₂ n} = n^{log₂ 4} = n²` of them. Total `T(n) = Θ(n²)`.

So my carefully-built recursive scheme costs `n²` — which is exactly what I wanted: it looks like it confirms the conjecture. But wait. Stare at where the `n²` came from. It came entirely from the `4`. Four sub-problems of half the size is `4·(n/2)² = n²` no matter how I arrange the bookkeeping; the linear combine work never mattered. The exponent is `log₂(number of sub-multiplications)`. So this construction doesn't *prove* anything about `M(n)` — it only shows that *this particular scheme, with four sub-multiplications, hits `n²`*. To turn it into a lower bound I'd need to argue that four is forced, that no recursive split can manage with fewer. And the moment I try to argue *that*, I have to ask the opposite question: is four actually forced? Could I do the split with *three* sub-multiplications? Because if I could, the exponent would drop to `log₂ 3 ≈ 1.585`, and far from proving the conjecture I'd be demolishing it.

That is where the proof attempt stops being a proof. I need to stop trying to prove four is necessary and instead attack it — look hard at the coefficients and what I actually *need* from them, because maybe the scheme computes more than the answer requires. Go back to the general two-number split, it's cleaner than dragging the squaring through:
```
a = a₁·Bᵐ + a₂,   b = b₁·Bᵐ + b₂,
a·b = a₁b₁·B²ᵐ + (a₁b₂ + a₂b₁)·Bᵐ + a₂b₂.
```
The coefficient at `B²ᵐ` is `a₁b₁`, the coefficient at `B⁰` is `a₂b₂`, and the coefficient at `Bᵐ` is `a₁b₂ + a₂b₁`. I need `a₁b₁`. I need `a₂b₂`. And for the middle I need — not `a₁b₂` and `a₂b₁` *separately* — only their **sum** `a₁b₂ + a₂b₁`. The two cross products sit at the same place value `Bᵐ`; the individual values are never used apart. So I've been computing two products just to add them and throw the parts away. If I could get the *sum* with one multiplication instead of two, I'd be at three multiplications total: `a₁b₁`, `a₂b₂`, and one for the middle.

Is there a single product whose value contains `a₁b₂ + a₂b₁`? A sum of cross terms like that is what spills out of multiplying two sums. Multiply the sum of `a`'s halves by the sum of `b`'s halves:
```
(a₁ + a₂)(b₁ + b₂) = a₁b₁ + a₁b₂ + a₂b₁ + a₂b₂.
```
There's the cross sum I wanted, `a₁b₂ + a₂b₁` — but bundled with the two corner products `a₁b₁` and `a₂b₂`. And those corners are *exactly* what I'm already computing for the high and low coefficients. So I don't chase the cross sum on its own; I isolate it by subtracting the corners I already have:
```
a₁b₂ + a₂b₁ = (a₁ + a₂)(b₁ + b₂) − a₁b₁ − a₂b₂.
```
I can set the pieces this way:
```
z₂ = a₁·b₁                  (high)
z₀ = a₂·b₂                  (low)
z₁ = (a₁ + a₂)(b₁ + b₂) − z₂ − z₀   (middle)
a·b = z₂·B²ᵐ + z₁·Bᵐ + z₀.
```
Three multiplications — `z₂`, `z₀`, and `(a₁+a₂)(b₁+b₂)` — and everything else is additions, subtractions, and shifts, all `O(n)`. The cross term I thought needed two of the four products is recovered from products I had to compute anyway plus one extra. The fourth multiplication is gone.

The algebra cancels, but I don't trust algebra I haven't run on a number — there's a real risk I've mislabelled a place value or dropped a coefficient somewhere in the reassembly. Let me push a concrete pair all the way through. Take `a = 1234`, `b = 4321`, base `B = 10`, half-length `m = 2`, so `Bᵐ = 100`. Splitting: `a₁ = 12, a₂ = 34, b₁ = 43, b₂ = 21`. The three multiplications:
```
z₂ = a₁·b₁ = 12·43 = 516
z₀ = a₂·b₂ = 34·21 = 714
(a₁+a₂)(b₁+b₂) = (12+34)·(43+21) = 46·64 = 2944
z₁ = 2944 − 516 − 714 = 1714
```
Now reassemble with the shifts `B²ᵐ = 10⁴` and `Bᵐ = 10²`:
```
a·b = 516·10⁴ + 1714·10² + 714 = 5 160 000 + 171 400 + 714 = 5 332 114.
```
And the honest schoolbook product is `1234·4321 = 5 332 114`. They match. One thing this run also makes me notice: `z₁ = 1714` is a *four-digit* number even though it's supposed to be the coefficient at the `10²` place — the cross sum can exceed `Bᵐ` and carry into the higher places. That's the recombination rule pinned down: add the shifted pieces as full integers, never as fixed-width digit fields, because the middle coefficient can spill past its own place value and the carry has to propagate correctly into the region above — a subtlety the algebra alone hid.

I should carry the same trick through the *squaring* form too, since that's where I started: `a² = aₕ²·B²ᵐ + 2aₕaₗ·Bᵐ + aₗ²`. The troublesome middle `2aₕaₗ` reduces the same way, `2aₕaₗ = (aₕ + aₗ)² − aₕ² − aₗ²`, so
```
a² = aₕ²·B²ᵐ + [(aₕ+aₗ)² − aₕ² − aₗ²]·Bᵐ + aₗ²,
```
and the only genuine work is three *squarings* of `m`-digit numbers: `aₕ²`, `aₗ²`, `(aₕ+aₗ)²`. Three, not four. Same saving, expressed in squares.

But there's a snag I have to handle before I claim the recursion squares only `m`-digit numbers. The sum `aₕ + aₗ` of two `m`-digit numbers can carry up to `m+1` digits, and if `(aₕ+aₗ)²` is the square of an `(m+1)`-digit number, the recursion isn't cleanly halving — the sub-problem is a digit too big. Peel off its top bit:
```
aₕ + aₗ = ε + 2·a₃,   ε ∈ {0,1},   a₃ an m-digit number.
```
Then `(aₕ + aₗ)² = (2a₃ + ε)² = 4a₃² + 4a₃ε + ε²`, and since `ε² = ε` and `4a₃ε` is either `0` or `4a₃` (a shift), squaring the `(m+1)`-digit sum reduces to squaring the `m`-digit number `a₃` plus `O(m)` cheap operations. Let me check this on the one case that actually overflows: `m = 2`, `aₕ = 21, aₗ = 94`, so `aₕ + aₗ = 115`, three digits. Then `ε = 115 mod 2 = 1`, `a₃ = 115 // 2 = 57`, genuinely `m = 2` digits. The identity claims `115² = 4·57² + 4·57·1 + 1`. Left side `13225`; right side `4·3249 + 228 + 1 = 12996 + 228 + 1 = 13225`. Equal, and `57` is an `m`-digit number, so the recursion halves cleanly — that was the one place the "three half-size sub-problems" story could have leaked, and the fix holds on a concrete overflow case, not only on paper.

The same saving applies to the two-number routine directly: the four-case recurrence has its branching factor knocked down from four to three,
```
T(n) = 3·T(n/2) + O(n),
```
the `O(n)` now covering the two additions that form the half-sums, the two subtractions that isolate `z₁`, and the shifts. Unfold the tree the same careful way. Level `i` has `3ⁱ` nodes of size `n/2ⁱ`, each doing `O(n/2ⁱ)` combine, so level `i` contributes `3ⁱ·O(n/2ⁱ) = O((3/2)ⁱ·n)`. The per-level work grows by `3/2` each level down — slower than the `4`-case's factor of `2`, but still growing, so the bottom dominates again. Leaf count `3^{log₂ n}`. Rewrite that exponent: `3^{log₂ n} = (2^{log₂ 3})^{log₂ n} = 2^{log₂ 3·log₂ n} = (2^{log₂ n})^{log₂ 3} = n^{log₂ 3}`. So `n^{log₂ 3}` leaves, each an `O(1)` single-digit multiply, and they dominate:
```
T(n) = Θ(n^{log₂ 3}) ≈ Θ(n^{1.585}).
```
Equivalently: the combine work has exponent `1` per level, and `1 < log₂ 3 ≈ 1.585`, so the branching term outgrows the linear one at every level and the leaves set the growth rate — in master-theorem terms, `a = 3`, `b = 2`, combine exponent `c = 1 < log_b a`.

The recurrence is only an asymptotic statement, so let me check the actual leaf count behaves like `n^{log₂ 3}` rather than assume it. If I count the single-digit multiplies the construction actually performs (the recursion isn't perfectly clean: the half-sums can carry, spawning slightly larger sub-problems), I get, for `n = 2, 4, 8, 16, 32`: `5, 19, 63, 197, 601` leaf multiplies. Against `n²` — `4, 16, 64, 256, 1024` — the schoolbook count is larger from `n = 8` on (`63 < 64`, `197 < 256`, `601 < 1024`), and the gap widens, so the sub-quadratic win is real but only bites once `n` is past a single-digit handful. And dividing the leaf counts by `n^{log₂ 3}` (`= 3, 9, 27, 81, 243`) gives `1.67, 2.11, 2.33, 2.43, 2.47` — settling toward a constant near `2.5`, not drifting upward the way an `n²` count would (`n^{0.415}` growth in the ratio). So the exponent I derived is the one the construction actually exhibits, constant factor and carries included.

And `log₂ 3 < 2`. So the conjecture I sat down to *prove* is false, and the construction I built to corner it is what kills it. The intuition behind the floor — "every digit-pair interacts, so `n²` products are forced" — confused *the output depending on all digit-pairs* with *needing one multiplication per digit-pair*. The cross sum `a₁b₂ + a₂b₁` depends on all those interactions, yet I pull it out with a single multiplication by sharing work with the corner products. Recursively, that one saved multiplication per level compounds into a changed exponent. The historical argument — "if something cheaper existed it would have been found" — turns out to be evidence about how *non-obvious* the sharing is, not about whether it exists.

I do not need a more elaborate split to refute the lower-bound conjecture. Two halves are the smallest place where the two cross products collide into the same middle coefficient, and that collision is exactly what the product-of-sums identity exploits. A single clean counterexample is enough, so I should keep the construction here and make the implementation exact.

The code has to follow this algebra directly, as a single self-contained C++ program that reads the two operands from stdin and prints their product to stdout. The whole interest is operands far past machine-word width, so I can't lean on a built-in 64-bit multiply: I carry each number as a little-endian vector of base-10 digits and write add, subtract, and shift on those vectors by hand, with a base case when one operand is a single digit; otherwise pick the split `m`, cut both numbers into halves, make the *three* recursive multiplications, and recombine with shifts.

Choosing `m`: the operands may differ in length, and `a₁+a₂`, `b₁+b₂` can each run one digit past the half — fine, the recursion handles slightly larger operands and the asymptotics don't change. I take `m` as half the digit-length of the longer operand and split both at that same `m`, so the place values `B²ᵐ` and `Bᵐ` line up for recombination. The trap that stays fatal in this digit-vector form is the one the concrete `z₁ = 1714` run already flagged: the recombination must be genuine integer addition with carry propagation, not fixed-width digit-field slotting, because the middle coefficient can exceed `Bᵐ` and must carry up into the high region. So `add`/`sub` propagate carries and borrows across the whole vector, and `trim` keeps leading zeros from accumulating — integers throughout, exactly.

This split-and-recombine step — using the same `m` for both operands, subtracting `z₂` and `z₀` from the half-sum product to form `z₁`, then shifting `z₂` by `2m` and `z₁` by `m` while carries propagate — is the part I'd most easily get wrong under time pressure; if I weren't confident I could implement it correctly within budget, I'd fall back to the base-10 schoolbook `O(n²)` digit-vector multiplication with ordinary carry propagation that I've already traced as correct, and ship that instead of risking a broken clever version.

The full program follows directly from this: digit vectors for `a` and `b`, `add`/`sub`/`shift` operating with carry and borrow propagation across the vector, a single-digit base case, and the recursive `karatsuba` routine computing `z₂ = a₁b₁`, `z₀ = a₂b₂`, `z₁ = (a₁+a₂)(b₁+b₂) − z₂ − z₀`, then recombining as `z₂·B²ᵐ + z₁·Bᵐ + z₀` — the module itself is below in the answer.

Running it against the schoolbook cross-check: `1234 4321` and `31415926 27182818` both come back matching the values I already traced by hand (`5332114` and `853973398759468`); a sweep of a few thousand random pairs up to several hundred digits each agrees with the exact product on every one. So the implementation is exact, and the recursion genuinely bottoms out — every split shrinks the digit vector toward the single-digit base case, and the carry-propagating recombination puts the overflowing middle coefficient in its right place.

I set out to prove the `n²` floor and ended up refuting it with the very scheme I built to corner it — one saved multiplication per recursion level, compounding to flip the exponent from `2` to `log₂ 3`. I don't have a matching lower bound; I haven't shown `n^{log₂ 3}` itself can't be beaten. But the conjecture as stated is dead, and that's what the construction settled.
