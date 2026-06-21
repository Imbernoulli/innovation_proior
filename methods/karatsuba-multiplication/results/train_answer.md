The problem is the oldest one in arithmetic: multiply two $n$-digit integers exactly, and do it cheaply. The only method in universal use is the schoolbook one — form a single-digit product $a_i b_j$ for every pair of digits, drop each into its place at position $i+j$, and sum the shifted rows with carries. With $n$ digits in each operand that is $n$ rows of $n$ single-digit multiplications: $\Theta(n^2)$ digit products plus $\Theta(n^2)$ additions. The trouble is purely the quadratic growth — doubling the operands quadruples the work, so for numbers with hundreds or thousands of digits the $n^2$ term swamps everything. The standing belief, stated as a conjecture around 1956 and again at the 1960 Moscow seminar, was that $M(n)$, the least number of bit operations sufficient to multiply $n$-digit numbers, is forced to grow like $n^2$ — that quadratic is a genuine floor. The arguments for it were two: a historical one (people have used the row method for four millennia, and if anything cheaper existed it would surely have been found) and a structural one (every digit of $a$ genuinely meets every digit of $b$ in the product, so it feels as though $n^2$ digit-interactions, and hence $n^2$ multiplications, cannot be avoided). The alternatives on the table did not touch this. Residue number systems (Svoboda–Valach) make multiplication cheap residue-by-residue but cannot even compare two numbers without an expensive conversion back, so they do not improve $M(n)$ at all. Bit-group splitting only trades operation count against depth, reorganizing the same quadratic digit work rather than removing it. And the obvious recursive idea — divide and conquer — buys nothing on its own: splitting each operand in half and recursing on the four half-size products gives $T(n) = 4\,T(n/2) + O(n)$, whose recursion tree has $4^{\log_2 n} = n^{\log_2 4} = n^2$ leaves, exactly the schoolbook cost, the exponent pinned at $\log_2 4 = 2$ by the count of four sub-multiplications.

That last observation is the lever, and what I propose is Karatsuba multiplication, which turns the four sub-multiplications into three and so breaks the quadratic floor. The starting point is the half-split: cut each operand at a $B^m$ boundary with $m \approx n/2$,
$$a = a_1 B^m + a_2, \qquad b = b_1 B^m + b_2,$$
so that the product expands into three place-value coefficients,
$$a\cdot b = a_1 b_1\, B^{2m} + (a_1 b_2 + a_2 b_1)\, B^m + a_2 b_2.$$
The naive reading sees four products — $a_1b_1$, $a_1b_2$, $a_2b_1$, $a_2b_2$ — but the decisive point is to stare at what the coefficients actually require. The high coefficient needs $a_1b_1$ and the low coefficient needs $a_2b_2$, both genuine products. The middle coefficient, however, needs only the *sum* $a_1b_2 + a_2b_1$, never the two cross products apart: they sit at the same place value $B^m$ and the individual values are thrown away after being added. So computing them as two separate multiplications is waste. A sum of cross terms like that is exactly what falls out of multiplying two sums, and the sum of the halves of each operand gives
$$(a_1 + a_2)(b_1 + b_2) = a_1 b_1 + (a_1 b_2 + a_2 b_1) + a_2 b_2.$$
The cross sum I want is bundled there with the two corner products $a_1b_1$ and $a_2b_2$ — and those corners are precisely the high and low coefficients I am already computing. So I do not chase the cross sum on its own; I isolate it by subtracting the corners I already have:
$$a_1 b_2 + a_2 b_1 = (a_1 + a_2)(b_1 + b_2) - a_1 b_1 - a_2 b_2.$$
This sets the three pieces of one split as
$$z_2 = a_1 b_1, \qquad z_0 = a_2 b_2, \qquad z_1 = (a_1 + a_2)(b_1 + b_2) - z_2 - z_0,$$
$$a\cdot b = z_2\, B^{2m} + z_1\, B^m + z_0.$$
Only three multiplications are recursive — $z_2$, $z_0$, and the product of half-sums — while everything else (the two subtractions, the additions, and the shifts by powers of $B$) is $O(n)$. Expanding $(a_1+a_2)(b_1+b_2) - z_2 - z_0$ collapses to $a_1b_2 + a_2b_1$ exactly, and reassembling recovers $a_1b_1 B^{2m} + (a_1b_2+a_2b_1)B^m + a_2b_2$ — the true product, not an approximation, which matters because the answer must be exact for every input. The reason this beats the four-product split is not bookkeeping cleverness at one level; it is that the saved multiplication compounds through the recursion. The same construction works in squaring form, which is equivalent up to a constant since $a\cdot b = \tfrac14[(a+b)^2 - (a-b)^2]$: writing $a = a_h B^m + a_l$ and using $2a_h a_l = (a_h+a_l)^2 - a_h^2 - a_l^2$ gives
$$a^2 = a_h^2\, B^{2m} + \big[(a_h+a_l)^2 - a_h^2 - a_l^2\big] B^m + a_l^2,$$
three squarings of $m$-digit numbers rather than four. One subtlety has to be handled for the recursion to halve cleanly: the half-sum $a_h + a_l$ can carry to $m+1$ digits. Peeling off its top bit as $a_h + a_l = 2a_3 + \varepsilon$ with $\varepsilon \in \{0,1\}$ gives $(a_h+a_l)^2 = 4a_3^2 + 4a_3\varepsilon + \varepsilon^2$, where $\varepsilon^2 = \varepsilon$ and $4a_3\varepsilon$ is either zero or a shift, so the square of the possibly-oversized sum reduces back to an $m$-digit square plus $O(m)$ cheap operations.

The payoff is read straight off the recurrence. With three half-size sub-multiplications and linear recombination,
$$T(n) = 3\,T(n/2) + O(n).$$
The recursion tree has $\log_2 n$ levels; level $i$ holds $3^i$ nodes of size $n/2^i$, each doing $O(n/2^i)$ combine work, so the level contributes $O\!\big((3/2)^i\, n\big)$. Because $3/2 > 1$ the per-level work grows downward and the leaves dominate: there are $3^{\log_2 n} = n^{\log_2 3}$ leaves, each a single-digit multiply, so
$$T(n) = \Theta\!\big(n^{\log_2 3}\big) \approx \Theta\!\big(n^{1.585}\big),$$
strictly sub-quadratic. (In master-theorem terms $a = 3$, $b = 2$, combine exponent $c = 1 < \log_2 3$, so the leaf term $n^{\log_b a}$ wins.) And $\log_2 3 < 2$, so the conjectured floor is false. The intuition behind the floor confused the product *depending* on all $n^2$ digit-pair interactions with *needing* one multiplication per pair: the cross sum $a_1b_2 + a_2b_1$ does depend on those interactions, yet it is pulled out with a single multiplication by sharing work with the corner products. Two halves are the smallest split where the two cross products collide into one middle coefficient, and that collision is exactly what the product-of-sums identity exploits — no more elaborate split is needed to refute the lower bound. The implementation follows the algebra directly: a base case when an operand is a single digit; otherwise pick $m$ as half the digit-length of the longer operand and split both at that same $m$ so the place values line up, make the three recursive multiplications, and recombine with shifts. The one language trap is fatal if missed — the split must use integer floor-division and remainder (`divmod`), because true division would turn the halves into floats, the operands would stop shrinking toward the base case, and the recursion would never bottom out.

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

    # cut each number into high/low halves at the B^m boundary.
    # integer floor-division + remainder (divmod) — true division would
    # turn the operands into floats and never reach the base case.
    high1, low1 = divmod(x, split)   # x = high1 * 10^m + low1
    high2, low2 = divmod(y, split)   # y = high2 * 10^m + low2

    # the THREE recursive multiplications
    z2 = karatsuba(high1, high2)                # a1 * b1
    z0 = karatsuba(low1, low2)                  # a2 * b2
    z3 = karatsuba(high1 + low1, high2 + low2)  # (a1+a2)(b1+b2)

    # middle coefficient: cross sum recovered from the product-of-sums
    # minus the two corner products already computed
    z1 = z3 - z2 - z0                           # = a1*b2 + a2*b1

    # recombine: x*y = z2 * B^(2m) + z1 * B^m + z0   (shifts + adds, O(n))
    return z2 * BASE ** (2 * m) + z1 * BASE ** m + z0
```
