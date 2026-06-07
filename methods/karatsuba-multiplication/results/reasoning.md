OK, let me think this through from scratch. The thing I'm actually after is a lower bound. Kolmogorov stated the conjecture at the seminar: multiplying two `n`-digit numbers needs on the order of `n²` bit operations, and that's a floor — no method can beat it. I find that very plausible and I want to *prove* it. The way you prove a lower bound like this is to corner the problem: build the most economical multiplication automaton you can, analyze exactly how its operations are forced, and show the count can't sink below `n²`. So my plan is to construct the tightest recursive multiplication scheme I can and then read off the lower bound from its structure. Let me build that scheme.

The only multiplication I actually know how to execute is the schoolbook one: take every digit of the first number, multiply it against every digit of the second, drop each little product into its place according to where the two digits sat, and add up the carries. If both numbers have `n` digits, that's `n` rows of `n` single-digit multiplications — `n²` one-digit multiplies, plus another `n²`-ish additions for the carries. Cost `Θ(n²)`. This is the thing the conjecture says is optimal, and the historical argument for it is strong: people have used this method for four thousand years, and if anything cheaper existed surely someone would have stumbled on it by now. The structural intuition is just as compelling — in the final product, *every* digit of `a` genuinely meets *every* digit of `b`; digit `aᵢ` lands at place `i+j` for each `bⱼ`, there are `n²` such pairs, and they all contribute. So it *feels* like `n²` little multiplications, one per pair, with no escape.

Before I try to turn that feeling into a proof, let me first simplify what I'm proving about. Multiplication and squaring are the same problem up to a constant, because
```
a·b = ¼[(a+b)² − (a−b)²],
```
and dividing by 4 in binary is just a two-bit shift. So `M(n)`, the complexity of multiplying two `n`-digit numbers, equals up to a constant the complexity of squaring one `n`-digit number — Kolmogorov pointed this out immediately. Good: I'll study squaring `a²`, one operand instead of two. The lower bound for squaring transfers to multiplication.

Now, to get a handle on `a²` recursively, I split `a` by digit position. The number is built out of its digits, so cut it at the half. With `n = 2m`, write
```
a = aₕ·Bᵐ + aₗ,
```
`aₕ` the high `m` digits, `aₗ` the low `m` digits. Square it:
```
a² = aₕ²·B²ᵐ + 2aₕaₗ·Bᵐ + aₗ².
```
Three pieces. Two of them, `aₕ²` and `aₗ²`, are squares of `m`-digit numbers — exactly the smaller version of the problem I'm recursing on. The middle piece `2aₕaₗ` is a *product*, not a square. If I insist on a square-only recursion, I would reduce that product with the same identity `ab = ¼[(a+b)²−(a−b)²]`, which means two more `m`-size squarings. If I stop using squares and analyze a two-operand product directly, the ordinary half-split has the same four products: high-high, high-low, low-high, low-low. So the naive recursive picture is four half-size multiplication problems plus linear recombination. Let me write the cost that way and see what lower bound it forces.

If squaring an `n`-digit number reduces to four `m`-size sub-multiplications plus `O(n)` shifts and adds, the cost obeys
```
T(n) = 4·T(n/2) + O(n).
```
Let me unfold the recursion tree, because the whole lower-bound argument lives in how this sum grows. At the root, `O(n)` combine-work. The root has 4 children of size `n/2`, each doing `O(n/2)` combine, so that level is `4·O(n/2) = O(2n)`. Next level: `4² = 16` nodes of size `n/4`, doing `16·O(n/4) = O(4n)`. Level `i` has `4ⁱ` nodes of size `n/2ⁱ`, contributing `4ⁱ·O(n/2ⁱ) = O(2ⁱ·n)`. The per-level work *doubles* going down, so the bottom dominates: there are `4^{log₂ n} = n^{log₂ 4} = n²` leaves, each a single-digit multiply. Total `T(n) = Θ(n²)`.

So my carefully-built recursive scheme costs `n²` — which is exactly what I wanted: it looks like it confirms the conjecture. But wait. Stare at where the `n²` came from. It came entirely from the `4`. Four sub-problems of half the size is `4·(n/2)² = n²` no matter how I arrange the bookkeeping; the linear combine work never mattered. The exponent is `log₂(number of sub-multiplications)`. So this construction doesn't *prove* anything about `M(n)` — it only shows that *this particular scheme, with four sub-multiplications, hits `n²`*. To turn it into a lower bound I'd need to argue that four is forced, that no recursive split can manage with fewer. And the moment I try to argue *that*, I have to ask the opposite question: is four actually forced? Could I do the split with *three* sub-multiplications? Because if I could, the exponent would drop to `log₂ 3 ≈ 1.585`, and far from proving the conjecture I'd be demolishing it.

That is where the proof attempt stops being a proof. I need to stop trying to prove four is necessary and instead attack it — look hard at the three coefficients and what I actually *need* from them, because maybe the scheme computes more than the answer requires. Go back to the general two-number split, it's cleaner than dragging the squaring through:
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
Three multiplications — `z₂`, `z₀`, and `(a₁+a₂)(b₁+b₂)` — and everything else is additions, subtractions, and shifts, all `O(n)`. The cross term I thought needed two of the four products is recovered from products I had to compute anyway plus one extra. The fourth multiplication is gone. Let me verify the middle by expanding:
```
(a₁+a₂)(b₁+b₂) − z₂ − z₀ = (a₁b₁ + a₁b₂ + a₂b₁ + a₂b₂) − a₁b₁ − a₂b₂ = a₁b₂ + a₂b₁. ✓
```
And reassembling gives back `a₁b₁·B²ᵐ + (a₁b₂+a₂b₁)·Bᵐ + a₂b₂`, the exact product — not an approximation, which matters because I need the precise answer.

I should also carry the same trick through the *squaring* form, since that's the version my recursion is actually set up on and I want the algebra airtight there. With `a = aₕ·Bᵐ + aₗ`,
```
a² = aₕ²·B²ᵐ + 2aₕaₗ·Bᵐ + aₗ².
```
The troublesome middle is `2aₕaₗ`, and the same identity gives it without a separate product:
```
2aₕaₗ = (aₕ + aₗ)² − aₕ² − aₗ².
```
So the square can be written as
```
a² = aₕ²·B²ᵐ + [(aₕ+aₗ)² − aₕ² − aₗ²]·Bᵐ + aₗ²,
```
and the only genuine work is three *squarings* of `m`-digit numbers: `aₕ²`, `aₗ²`, `(aₕ+aₗ)²`. Three, not four. Same saving, expressed in squares.

But there's a snag I have to handle before I claim the recursion squares only `m`-digit numbers. The sum `aₕ + aₗ` of two `m`-digit numbers can carry up to `m+1` digits. If `(aₕ+aₗ)²` is the square of an `(m+1)`-digit number, the recursion isn't cleanly halving — the sub-problem is a digit too big. Let me absorb that overflow. Write the possibly-`(m+1)`-digit sum by peeling off its top bit:
```
aₕ + aₗ = ε + 2·a₃,   ε ∈ {0,1},   a₃ an m-digit number.
```
Then
```
(aₕ + aₗ)² = (2a₃ + ε)² = 4a₃² + 4a₃ε + ε²,
```
and `ε² = ε`, `4a₃ε` is either `0` or `4a₃` (a shift), so squaring the `(m+1)`-digit sum reduces to squaring the `m`-digit number `a₃` plus `O(m)` cheap operations. So the three sub-squarings are genuinely all of `m`-digit numbers, and the recursion halves cleanly. Good — that was the one place the clean "three half-size sub-problems" story could have leaked, and it's patched with a shift and an add.

Let me make the cost precise, keeping the digit-length variable straight. If squaring an `r`-digit number takes `N(r)` operations, then squaring a `2r`-digit number — three `r`-digit squarings plus shifts and adds — takes
```
N(2r) ≤ 3·N(r) + c·r,
```
for a constant `c`: the `3·N(r)` is the three sub-squarings, and the `c·r` is the linear combine (the shifts, the algebraic sum of at most seven `O(r)`-digit numbers, and the overflow fix). If I index the lengths as `1, 2, 4, ...`, the same statement appears as `N_{q+1} ≤ 3N_q + c·2^q`; the exponential is only because the digit length at level `q` is `2^q`. Unrolling over `log₂ n` levels gives a sum where the branching by 3 outgrows the linear combine terms. The dominant term is the `3^{log₂ n}` from the leaves.

For the two-number multiplication routine, the four-case recurrence has the `4` knocked down to `3`:
```
T(n) = 3·T(n/2) + O(n).
```
Unfold the tree the same careful way. Level `i` has `3ⁱ` nodes of size `n/2ⁱ`, each doing `O(n/2ⁱ)` combine, so level `i` contributes `3ⁱ·O(n/2ⁱ) = O((3/2)ⁱ·n)`. The per-level work grows by `3/2` each level down — slower than the `4`-case's factor of `2`, but still growing, so the bottom dominates again. Leaf count `3^{log₂ n}`. Rewrite that exponent: `3^{log₂ n} = (2^{log₂ 3})^{log₂ n} = 2^{log₂ 3·log₂ n} = (2^{log₂ n})^{log₂ 3} = n^{log₂ 3}`. So `n^{log₂ 3}` leaves, each an `O(1)` single-digit multiply, and they dominate:
```
T(n) = Θ(n^{log₂ 3}) ≈ Θ(n^{1.585}).
```
Let me confirm which term dominates by summing the geometric series outright: total `= ∑_{i=0}^{log₂ n} (3/2)ⁱ·n = n·∑(3/2)ⁱ`. Since `3/2 > 1`, the sum is `Θ((3/2)^{log₂ n})`, so total `= Θ(n·(3/2)^{log₂ n})`. And `(3/2)^{log₂ n} = 3^{log₂ n}/2^{log₂ n} = n^{log₂ 3}/n`, so `n·n^{log₂ 3}/n = n^{log₂ 3}`. Same answer, and now I see exactly why: the linear combine work per level, with exponent `1`, is strictly below `log₂ 3 ≈ 1.585`, so the branching out-grows it and the leaves set the rate.

And `log₂ 3 < 2`. So the conjecture I sat down to *prove* is false, and the construction I built to corner it is what kills it. The intuition behind the floor — "every digit-pair interacts, so `n²` products are forced" — confused *the output depending on all digit-pairs* with *needing one multiplication per digit-pair*. The cross sum `a₁b₂ + a₂b₁` depends on all those interactions, yet I pull it out with a single multiplication by sharing work with the corner products. Recursively, that one saved multiplication per level compounds into a changed exponent. The historical argument — "if something cheaper existed it would have been found" — turns out to be evidence about how *non-obvious* the sharing is, not about whether it exists.

I do not need a more elaborate split to refute the lower-bound conjecture. Two halves are the smallest place where the two cross products collide into the same middle coefficient, and that collision is exactly what the product-of-sums identity exploits. A single clean counterexample is enough, so I should keep the construction here and make the implementation exact.

The code has to follow this algebra directly, with a base case when an operand is a single digit; otherwise pick the split `m`, cut both numbers into halves, make the *three* recursive multiplications, and recombine with shifts.

Choosing `m`: the operands may differ in length, and `a₁+a₂`, `b₁+b₂` can each run one digit past the half — fine, the recursion handles slightly larger operands and the asymptotics don't change. I take `m` as half the digit-length of the longer operand and split both at that same `m`, so the place values `B²ᵐ` and `Bᵐ` line up for recombination. The other trap is a language one but fatal: when I split, I must use *integer* floor-division and remainder; true division would turn the halves into floats, the operands would stop shrinking toward the single-digit base case, and the recursion would never bottom out. So `divmod` everywhere, integers throughout.

```python
BASE = 10

def karatsuba(x, y):
    # base case: a single-digit operand — multiply directly (O(1))
    if x < BASE or y < BASE:
        return x * y

    # split point: half the digit-length of the longer operand
    n = max(len(str(x)), len(str(y)))
    m = n // 2
    split = BASE ** m

    # cut each number into high and low halves at the B^m boundary.
    # integer floor-division + remainder — NOT true division, or the
    # operands turn into floats and never reach the base case.
    high1, low1 = divmod(x, split)   # x = high1 * 10^m + low1
    high2, low2 = divmod(y, split)   # y = high2 * 10^m + low2

    # the THREE multiplications — the whole point:
    z2 = karatsuba(high1, high2)                 # a1 * b1   (high coeff)
    z0 = karatsuba(low1, low2)                   # a2 * b2   (low coeff)
    z3 = karatsuba(high1 + low1, high2 + low2)   # (a1+a2)(b1+b2)

    # middle coeff = cross sum, recovered from the product-of-sums
    # minus the two corner products already computed:
    #   a1*b2 + a2*b1 = z3 - z2 - z0
    z1 = z3 - z2 - z0

    # recombine with shifts (powers of the base) and adds — all O(n):
    #   x*y = z2 * B^(2m) + z1 * B^m + z0
    return z2 * BASE ** (2 * m) + z1 * BASE ** m + z0
```

The causal chain, start to finish: I set out to *prove* the `n²` floor by building the tightest recursive multiplication scheme and reading the lower bound off it; the natural scheme splits each number in half and needs four sub-multiplications, giving `4T(n/2)+O(n) = n²` — which looked like confirmation until I saw the `n²` was nothing but `log₂ 4`, an artifact of the count four, not anything forced; staring at the three place-value coefficients showed the middle one needs only the *sum* of the two cross products, and that sum is exactly what `(a₁+a₂)(b₁+b₂)` carries minus the two corner products already in hand — so the cross sum costs one multiplication, not two; three sub-multiplications turns the recurrence into `3T(n/2)+O(n)`, whose tree has `n^{log₂ 3}` leaves; so multiplication runs in `Θ(n^{log₂ 3}) ≈ Θ(n^{1.585})`, and the conjecture I meant to confirm is refuted by the very construction meant to confirm it.
