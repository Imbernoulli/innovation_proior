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

The deliverable is a single self-contained C++17 program: it reads two non-negative big integers (whitespace-separated, arbitrary length) from `stdin` and prints their exact product to `stdout`. Because the whole point is operands far past machine-word width, the numbers are carried as little-endian base-10 digit vectors and all of add/sub/shift are done on those vectors; the recursion is the three-multiplication split above, with the split point `m` taken as half the longer operand's digit length. The half-sums `x1+x2`, `y1+y2` may carry an extra digit — that is harmless, the recursion simply takes a slightly longer sub-operand and the asymptotics are unchanged.

```cpp
// Karatsuba multiplication. Reads two non-negative big integers (whitespace-
// separated, arbitrary length) from stdin and prints their exact product.
#include <bits/stdc++.h>
using namespace std;

// A big number is held little-endian, one base-10 digit per vector slot.
using Big = vector<int>;          // digits[0] is the units place

static const int BASE = 10;

Big from_string(const string& s) {
    Big d;
    for (int i = (int)s.size() - 1; i >= 0; --i) d.push_back(s[i] - '0');
    if (d.empty()) d.push_back(0);
    return d;
}

void trim(Big& a) {                // drop leading (high-order) zeros
    while (a.size() > 1 && a.back() == 0) a.pop_back();
}

// add: a + b
Big add(const Big& a, const Big& b) {
    Big c;
    int carry = 0;
    for (size_t i = 0; i < a.size() || i < b.size() || carry; ++i) {
        int s = carry;
        if (i < a.size()) s += a[i];
        if (i < b.size()) s += b[i];
        c.push_back(s % BASE);
        carry = s / BASE;
    }
    return c;
}

// sub: a - b, assuming a >= b (used only where the algebra guarantees it)
Big sub(const Big& a, const Big& b) {
    Big c;
    int borrow = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int s = a[i] - borrow - (i < b.size() ? b[i] : 0);
        if (s < 0) { s += BASE; borrow = 1; } else borrow = 0;
        c.push_back(s);
    }
    trim(c);
    return c;
}

// shift: multiply by BASE^k (append k low-order zeros)
Big shift(const Big& a, size_t k) {
    if (a.size() == 1 && a[0] == 0) return a;   // 0 stays 0
    Big c(k, 0);
    c.insert(c.end(), a.begin(), a.end());
    return c;
}

bool is_zero(const Big& a) { return a.size() == 1 && a[0] == 0; }

// Karatsuba: three half-size multiplications instead of four.
//   z2 = x1*y1,  z0 = x2*y2,  z1 = (x1+x2)(y1+y2) - z2 - z0
//   x*y = z2*B^(2m) + z1*B^m + z0
Big karatsuba(const Big& x, const Big& y) {
    // base case: a single-digit operand -> multiply digit-by-number, O(len)
    if (x.size() == 1 || y.size() == 1) {
        long long mul = (x.size() == 1) ? x[0] : y[0];
        const Big& big = (x.size() == 1) ? y : x;
        Big c;
        long long carry = 0;
        for (size_t i = 0; i < big.size(); ++i) {
            long long s = (long long)big[i] * mul + carry;
            c.push_back((int)(s % BASE));
            carry = s / BASE;
        }
        while (carry) { c.push_back((int)(carry % BASE)); carry /= BASE; }
        if (c.empty()) c.push_back(0);
        trim(c);
        return c;
    }

    // split both operands at m = half the length of the longer one
    size_t m = max(x.size(), y.size()) / 2;
    size_t mx = min(m, x.size()), my = min(m, y.size());

    Big x2(x.begin(), x.begin() + mx);   // low  half of x
    Big x1(x.begin() + mx, x.end());     // high half of x
    Big y2(y.begin(), y.begin() + my);   // low  half of y
    Big y1(y.begin() + my, y.end());     // high half of y
    if (x1.empty()) x1.push_back(0);
    if (x2.empty()) x2.push_back(0);
    if (y1.empty()) y1.push_back(0);
    if (y2.empty()) y2.push_back(0);
    trim(x1); trim(x2); trim(y1); trim(y2);

    Big z2 = karatsuba(x1, y1);                       // a1*b1   (high)
    Big z0 = karatsuba(x2, y2);                       // a2*b2   (low)
    Big z3 = karatsuba(add(x1, x2), add(y1, y2));     // (a1+a2)(b1+b2)
    Big z1 = sub(sub(z3, z2), z0);                    // = a1*b2 + a2*b1

    // recombine: z2*B^(2m) + z1*B^m + z0
    Big result = add(add(shift(z2, 2 * m), shift(z1, m)), z0);
    trim(result);
    return result;
}

string to_string_big(const Big& a) {
    string s;
    for (int i = (int)a.size() - 1; i >= 0; --i) s += char('0' + a[i]);
    return s;
}

int main() {
    string sa, sb;
    if (!(cin >> sa >> sb)) return 0;
    Big a = from_string(sa), b = from_string(sb);
    trim(a); trim(b);
    Big prod = (is_zero(a) || is_zero(b)) ? Big{0} : karatsuba(a, b);
    cout << to_string_big(prod) << "\n";
    return 0;
}
```
