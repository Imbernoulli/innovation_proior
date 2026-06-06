OK, let me think this through from scratch. I have two integers, each up to `n` digits, and I want their product. The only method I actually know how to execute is the one from school: take every digit of the first number, multiply it against every digit of the second, drop each little product into its place according to where the two digits sat, and add up all the carries. If both numbers have `n` digits, that's `n` rows, each with `n` single-digit multiplications — `n²` one-digit multiplies, and another `n²`-ish additions to fold the carries together. The cost is `Θ(n²)`.

And the received wisdom is that you can't do better — that any multiplication procedure must, in the worst case, do on the order of `n²` elementary operations. I should take that belief seriously, because it has a very plausible story behind it: in the final product, *every* digit of `x` genuinely interacts with *every* digit of `y`. Digit `xᵢ` lands at place `i + j` for each `yⱼ`; there are `n²` such pairs, and they all contribute. So it *feels* like there must be `n²` little multiplications, one per pair, and you can't escape them.

But let me stare at that argument, because it's doing the work and I'm not sure it's airtight. The claim "every pair of digits contributes" is true. The claim "therefore I must *separately multiply* every pair of digits" is a leap. What actually has to be true is that the final digits of the answer depend on all the input digits. That's a statement about the *output*, not about *how many multiplications* I'm forced to do. Lots of things depend on all their inputs but can be computed with far less than one operation per input-pair. So the `n²` feeling might be an artifact of the schoolbook *layout*, not a law about the problem. Let me try to actually attack it rather than believe it.

The handle I have is divide-and-conquer: when a problem is built out of smaller copies of itself, split it. A number is built out of its digits, so the obvious split is to cut each number in half. Pick `m ≈ n/2`, and write each number in base `B` with its top half and bottom half separated at the `Bᵐ` boundary:

```
x = x₁·Bᵐ + x₀
y = y₁·Bᵐ + y₀
```

where `x₁` is the high `m` digits, `x₀` the low `m` digits, same for `y`. Now multiply, just expanding the product of the two binomials:

```
x·y = (x₁·Bᵐ + x₀)(y₁·Bᵐ + y₀)
    = x₁y₁·B²ᵐ + x₁y₀·Bᵐ + x₀y₁·Bᵐ + x₀y₀
    = x₁y₁·B²ᵐ + (x₁y₀ + x₀y₁)·Bᵐ + x₀y₀.
```

The multiplications by `B²ᵐ` and `Bᵐ` aren't real multiplications — they're just shifts, sliding the digits left, which costs `O(n)`. The genuine work is the four products `x₁y₁`, `x₁y₀`, `x₀y₁`, `x₀y₀`, each of two `m`-digit numbers, and then a couple of `O(n)` additions to combine. So I've turned one `n`-digit multiplication into four `(n/2)`-digit multiplications plus linear overhead. If I keep recursing — multiply the halves by splitting *them* in half, and so on down to single digits — the total cost obeys

```
T(n) = 4·T(n/2) + O(n).
```

Let me solve this, because the whole question is whether the recursion bought me anything. Unfold the recursion tree. At the root I do `O(n)` combine-work. The root has 4 children, each a problem of size `n/2`, each doing `O(n/2)` combine-work — so that level does `4 · O(n/2) = O(2n)`. The next level has `4² = 16` nodes of size `n/4`, doing `16 · O(n/4) = O(4n)`. In general level `i` has `4ⁱ` nodes each of size `n/2ⁱ`, contributing `4ⁱ · O(n/2ⁱ) = O(2ⁱ·n)`. The per-level work *doubles* as I go down. There are `log₂ n` levels, and the last level dominates: at the bottom there are `4^{log₂ n} = n^{log₂ 4} = n²` leaves, each a single-digit multiply. Summing, `T(n) = Θ(n²)`.

Wall. The recursion gave me *nothing*. I split, I recursed, I paid all the overhead of the divide-and-conquer machinery — and I landed right back on `n²`. And in hindsight that's obvious: four sub-problems of half the size is `4 · (n/2)² = n²` work at the leaves no matter how I shuffle it. The `4` is the killer. With `a` sub-multiplications of half-size numbers, the leaf count is `a^{log₂ n} = n^{log₂ a}`, and `a = 4` gives the exponent `log₂ 4 = 2`. So splitting in half is fine, but *as long as I keep doing four multiplications per split, I'm pinned at `n²`*. The believers were right about this particular method. To beat `n²` I have to drive that `4` down. If I could do the same split with only `3` sub-multiplications, the exponent would drop to `log₂ 3 ≈ 1.585`, which is genuinely sub-quadratic. So the real question sharpens: can I get `x·y` out of *three* products of half-size numbers instead of four?

Let me look hard at the four products and what I actually *need* from them, because maybe I'm computing more than the answer requires. The product is

```
x·y = x₁y₁·B²ᵐ + (x₁y₀ + x₀y₁)·Bᵐ + x₀y₀.
```

Look at the three coefficients sitting at the three place-values `B²ᵐ`, `Bᵐ`, `B⁰`:
- the high coefficient is `x₁y₁`,
- the low coefficient is `x₀y₀`,
- the middle coefficient is `x₁y₀ + x₀y₁`.

So I need `x₁y₁`, I need `x₀y₀`, and for the middle I need... not `x₁y₀` and `x₀y₁` *separately* — I only need their **sum** `x₁y₀ + x₀y₁`. That's the crack. I've been computing two products, `x₁y₀` and `x₀y₁`, just to add them together and throw away the individual values. They never appear apart in the answer; they sit at the same place value `Bᵐ`. If I could get the *sum* `x₁y₀ + x₀y₁` with a single multiplication instead of two, I'd be down to three multiplications total: `x₁y₁`, `x₀y₀`, and one more for the middle.

So: is there a single product whose value contains `x₁y₀ + x₀y₁`? A sum of cross terms like that smells like what falls out of multiplying two sums. Multiply the sum of the halves of `x` by the sum of the halves of `y`:

```
(x₁ + x₀)(y₁ + y₀) = x₁y₁ + x₁y₀ + x₀y₁ + x₀y₀.
```

There it is — the right-hand side contains the cross sum `x₁y₀ + x₀y₁` I wanted, but it's bundled together with the two corner products `x₁y₁` and `x₀y₀`. And those two corners are *exactly the products I'm already computing* for the high and low coefficients. So I don't need to chase the cross sum on its own; I can isolate it by subtracting the corners I already have:

```
x₁y₀ + x₀y₁ = (x₁ + x₀)(y₁ + y₀) − x₁y₁ − x₀y₀.
```

Let me name the three real multiplications and check the whole thing closes:

```
z₂ = x₁ · y₁            (high product)
z₀ = x₀ · y₀            (low product)
z₃ = (x₁ + x₀)(y₁ + y₀) (product of the sums)
```

Then the middle coefficient is `z₃ − z₂ − z₀`, and the answer is

```
x·y = z₂·B²ᵐ + (z₃ − z₂ − z₀)·Bᵐ + z₀.
```

Three multiplications — `z₂`, `z₀`, `z₃` — and everything else is additions, subtractions, and shifts, all `O(n)`. The cross term I thought required two of the four products is recovered *for free* from products I had to compute anyway plus one extra. The fourth multiplication is gone.

Let me sanity-check the algebra by expanding `z₃ − z₂ − z₀`:
```
z₃ − z₂ − z₀ = (x₁y₁ + x₁y₀ + x₀y₁ + x₀y₀) − x₁y₁ − x₀y₀ = x₁y₀ + x₀y₁. ✓
```
Exactly the middle coefficient. And let me check the whole product reassembles correctly:
```
z₂·B²ᵐ + (z₃ − z₂ − z₀)·Bᵐ + z₀
  = x₁y₁·B²ᵐ + (x₁y₀ + x₀y₁)·Bᵐ + x₀y₀,
```
which is the expansion I started from. So it's exact, not an approximation — good, because the problem demands the precise product.

Now there's a flicker of recognition. A product of two two-part things — `(x₁ + x₀)(y₁ + y₀)` — where you want the cross term but you compute it from one product-of-sums minus the two corner products... I've met this shape before. Multiplying two complex numbers `(a + b·i)(c + d·i)` gives real part `ac − bd` and imaginary part `ad + bc`; the imaginary part is a cross sum `ad + bc`, and the old trick is `ad + bc = (a + b)(c + d) − ac − bd`, so you get the complex product with three real multiplications instead of four. It's the same identity. The structural fact is that the cross term of a product of sums is recoverable from the product-of-sums minus the corner products — and here that fact is what cracks `n²`. What was a parlor trick for saving one real multiplication in a `2×2` bilinear form becomes, applied *recursively* to the digit-splitting, a change in the asymptotic exponent.

Let me ground it on a concrete pair before I trust the recursion: `1234 × 4321`. Four digits, split at `m = 2`, base `B = 10`. So `x₁ = 12, x₀ = 34, y₁ = 43, y₀ = 21`.
```
z₂ = 12 · 43 = 516
z₀ = 34 · 21 = 714
z₃ = (12 + 34)(43 + 21) = 46 · 64 = 2944
middle = z₃ − z₂ − z₀ = 2944 − 516 − 714 = 1714
x·y = 516·10⁴ + 1714·10² + 714 = 5 160 000 + 171 400 + 714 = 5 332 114.
```
And `1234 × 4321 = 5 332 114` directly. It checks. And notice the recursion is real: those three sub-products `12·43`, `34·21`, `46·64` are themselves multiplications of smaller numbers, which I'd compute the same way until the operands are single digits and I just multiply them outright.

Now the payoff — the cost. The recurrence is the same shape as before but with the `4` replaced by a `3`:
```
T(n) = 3·T(n/2) + O(n).
```
The `O(n)` is the combine work: forming `x₁ + x₀` and `y₁ + y₀`, the two subtractions for the middle, and the shifted additions — all linear. Let me unfold the tree the same careful way. Level `i` has `3ⁱ` nodes, each of size `n/2ⁱ`, each doing `O(n/2ⁱ)` combine work, so level `i` contributes `3ⁱ · O(n/2ⁱ) = O((3/2)ⁱ · n)`. Now the per-level work grows by a factor `3/2` each level down — *slower* than the `4`-case's factor of `2`, but still growing, so again the bottom level dominates. There are `log₂ n` levels, and the number of leaves is `3^{log₂ n}`. Rewrite that exponent: `3^{log₂ n} = (2^{log₂ 3})^{log₂ n} = 2^{log₂ 3 · log₂ n} = (2^{log₂ n})^{log₂ 3} = n^{log₂ 3}`. So there are `n^{log₂ 3}` leaves, each an `O(1)` single-digit multiply, and the total is dominated by them:
```
T(n) = Θ(n^{log₂ 3}) ≈ Θ(n^{1.585}).
```

Let me make sure I'm not fooling myself about which term dominates, by summing the geometric series explicitly. The total work is `∑_{i=0}^{log₂ n} (3/2)ⁱ · n = n · ∑ (3/2)ⁱ`. Since `3/2 > 1`, the sum is dominated by its largest term, `(3/2)^{log₂ n}`, up to a constant factor: `∑ (3/2)ⁱ = Θ((3/2)^{log₂ n})`. So `T(n) = Θ(n · (3/2)^{log₂ n})`. Now `(3/2)^{log₂ n} = 3^{log₂ n} / 2^{log₂ n} = n^{log₂ 3} / n`, so `n · (3/2)^{log₂ n} = n · n^{log₂ 3}/n = n^{log₂ 3}`. Same answer, and now I see exactly why: the linear combine work per level is *out-grown* by the branching, so it's the leaves, not the root, that set the rate. This is the regime where the number of sub-problems dominates the per-level overhead — the `f(n) = O(n)` combine cost has exponent `1`, which is strictly below `log₂ 3 ≈ 1.585`, so the leaves win and the closed form is `n^{log₂ 3}`.

And that's strictly below `2`. The believed lower bound — that multiplication needs `Ω(n²)` operations — is just false. The intuition behind it ("every pair of digits interacts, so `n²` products are forced") confused *the output depending on all digit-pairs* with *needing one multiplication per digit-pair*. The cross term `x₁y₀ + x₀y₁` depends on all four half-products' worth of digit interactions, yet I extract it with a single multiplication by sharing work with the corner products. Recursively, that single saved multiplication per level compounds into a changed exponent.

Let me also pin down why I split into *two* halves and not, say, three or more pieces, since the trick was really about reducing the multiplication count. The leaf count is `a^{log_b n} = n^{log_b a}` when I split each number into `b` pieces and manage with `a` sub-multiplications. Splitting in half (`b = 2`) and getting it down to `a = 3` gives exponent `log₂ 3`. The principle generalizes — split into more pieces and you'd want to drive `a` even lower relative to `b²`, which is a direction worth chasing later — but the two-piece, three-multiplication version already breaks the barrier, so that's the one to nail down now.

Now let me turn this into code, and stay honest about the places where a naive transcription bites. The structure is: base case when an operand is a single digit; otherwise pick the split point `m`, cut both numbers into high/low halves, make the *three* recursive multiplications, recombine with shifts.

There's a subtlety in choosing `m`. The two operands may differ in length, and after splitting, `x₁ + x₀` and `y₁ + y₀` can each be one digit longer than the half — that's fine, the recursion just handles slightly larger operands, and it doesn't change the asymptotics. I'll take `m` as half the digit-length of the longer operand and split both at that same `m`, so the place values `B²ᵐ` and `Bᵐ` line up consistently for the recombination.

The other trap is purely a language one but it's fatal: when I split with `m = n // 2` I'm using integer division, and I must split the numbers with *integer* floor-division and remainder too. If I write `x / 10**m` with true division I get a float, the operands stop shrinking toward the single-digit base case in integer-land, and the recursion never bottoms out. So: floor-divide and remainder (`divmod`) everywhere, integers throughout.

```python
def karatsuba(x, y):
    # base case: a single-digit operand — just multiply directly (O(1))
    if x < 10 or y < 10:
        return x * y

    # split point: half the digit-length of the longer operand
    n = max(len(str(x)), len(str(y)))
    m = n // 2
    split = 10 ** m

    # cut each number into high and low halves at the B^m boundary.
    # integer floor-division + remainder — NOT true division, or the
    # operands turn into floats and never reach the base case.
    high1, low1 = divmod(x, split)   # x = high1 * 10^m + low1
    high2, low2 = divmod(y, split)   # y = high2 * 10^m + low2

    # the THREE multiplications — the whole point.
    z0 = karatsuba(low1, low2)                       # x0 * y0   (low coeff)
    z2 = karatsuba(high1, high2)                     # x1 * y1   (high coeff)
    z3 = karatsuba(high1 + low1, high2 + low2)       # (x1+x0)(y1+y0)

    # middle coeff = cross sum, recovered from the product-of-sums
    # minus the two corner products I already have:
    #   x1*y0 + x0*y1 = z3 - z2 - z0
    z1 = z3 - z2 - z0

    # recombine with shifts (powers of the base) and adds — all O(n):
    #   x*y = z2 * B^(2m) + z1 * B^m + z0
    return z2 * 10 ** (2 * m) + z1 * 10 ** m + z0
```

The causal chain, start to finish: schoolbook multiplication costs `n²` because it forms a separate product for every digit-pair, and the folklore said that was unavoidable; splitting each number in half and recursing naively keeps four sub-multiplications and so reproduces exactly `n²` — the recursion alone buys nothing, because four half-size problems is still `n²`; staring at the three place-value coefficients shows the middle one needs only the *sum* of the two cross products, not the products themselves; that sum is exactly what `(x₁+x₀)(y₁+y₀)` carries, minus the two corner products `x₁y₁` and `x₀y₀` that are already being computed — so the cross sum costs *one* multiplication instead of two; three sub-multiplications per split turns the recurrence `4T(n/2)+O(n)` into `3T(n/2)+O(n)`, whose recursion tree has `n^{log₂ 3}` leaves; and so the whole thing runs in `Θ(n^{log₂ 3}) ≈ Θ(n^{1.585})`, breaking the `n²` barrier that everyone believed was a law.
