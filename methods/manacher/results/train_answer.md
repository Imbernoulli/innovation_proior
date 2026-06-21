The task is to take a string $s$ of length $n$ and return one longest substring that reads the same forwards and backwards, and to do it in $O(n)$ time while exposing reusable per-center data for callers that want more than the single answer. The natural way to organize palindromes is by center: every palindrome either sits on a real character, like `aba`, or straddles a gap, like `abba`. So I can expand around each character and each gap, comparing the two outward characters and stopping at the first mismatch or boundary. That is correct, but the cost is fatal. On an input like `"aaaa...a"`, almost every center expands nearly to the ends, so the same character equalities are rediscovered over and over; with linearly many centers each paying up to linearly many comparisons, the direct expansion is $\Theta(n^2)$. The waste has a precise shape that points to its own cure: when expanding one center reveals a long palindrome, I have learned that a whole interval is mirror-symmetric about that center, and the next center inside that interval is not starting from blank space — its left neighborhood is already a reflection of what lies to its right.

What I propose is Manacher's algorithm, which converts that observation into a single left-to-right scan that touches each position a constant number of times. The idea is to carry, at all times, the one palindrome discovered so far whose right edge reaches farthest, because that is the palindrome whose symmetry can help the largest remaining suffix of unprocessed centers. I keep its center $c$ and its inclusive right boundary $r$, so the carried palindrome covers indices $2c - r$ through $r$. When I arrive at a new center $i$, there are two cases. If $i \ge r$, $i$ lies outside the carried interval, the symmetry tells me nothing, and I begin with radius zero. If $i < r$, then $i$ has a reflected center $\text{mirror} = 2c - i$ whose radius is already computed, and by the symmetry of the carried palindrome the radius around $\text{mirror}$ transfers to $i$ — but only as far as it stays inside the carried interval, because the character just past $r$ has never been compared and the old symmetry says nothing about it. The largest radius I may safely assume without any new work is therefore

$$p[i] = \min(r - i,\; p[\text{mirror}]).$$

The order of these two operations is the load-bearing design choice: seed from the mirror, clamp to the current right edge, and only then resume direct expansion from the first pair that has not yet been verified. Clamping before expanding is what keeps the bound honest — if I instead trusted $p[\text{mirror}]$ unclamped, I could claim symmetry past $r$ that has never been checked and get a wrong answer; if I clamped but re-checked from scratch inside the interval, I would throw away exactly the savings that make the method linear. After the seed, the next attempted comparison is at indices $i - p[i] - 1$ and $i + p[i] + 1$; while those are in bounds and equal, I increment $p[i]$, and otherwise the radius is final. Whenever the new palindrome reaches farther right than anything before it, i.e. $i + p[i] > r$, I replace the carried pair with $c = i$ and $r = i + p[i]$.

The linearity follows from an amortized argument on the right boundary. After the clamp, any center inside the carried interval satisfies $i + p[i] \le r$, so its very first real comparison happens no earlier than the current frontier. A successful comparison can only verify a pair at or beyond that frontier, and each success pushes the right edge of the current center one position farther; since the carried $r$ never moves backward and the frontier can advance only across the whole transformed array, all successful comparisons summed over all centers are $O(n)$. Each center additionally pays at most one terminating failed comparison, and there are only $O(n)$ centers, with constant bookkeeping per center — so the entire pass is $O(n)$, using $O(n)$ extra space for the transformed sequence and the radius array.

That leaves the odd/even nuisance: palindromes centered on a character versus on a gap. Maintaining two near-identical routines would duplicate the whole mirror argument with shifted indices, so instead I fold both into one pass with a separator transform. Conceptually I weave a separator `#` between every pair of characters and at both ends, turning `abba` into `#a#b#b#a#`. Now an even-length palindrome of the original string becomes an odd palindrome centered on a separator, while an odd palindrome stays centered on its character, so every palindrome I ever consider in the transformed sequence is odd-centered and the single mirror pass handles them all. One subtlety: a literal `#` is only a safe separator if it cannot appear in the input, and Python strings can contain `#`, so the implementation uses a private `object()` as the separator inside a list — no input character can compare equal to it — while the derivation still reads as if it were `#`. Real characters land at odd transformed indices and separators at even ones, alternating, which gives the radius its clean meaning: because the two kinds of position strictly alternate, the number of original characters spanned by a transformed palindrome of radius $p[i]$ is exactly $p[i]$. So `#b#b#` centered on its middle separator has radius $2$ and original length $2$; `#a#` has radius $1$ and length $1$; a bare separator has radius $0$, the empty palindrome. The longest original length is thus $\max_i p[i]$. Recovering the substring uses the same layout: original character $s[j]$ sits at transformed index $2j + 1$, so if $\text{best\_center}$ maximizes $p$ and $\text{best\_len} = p[\text{best\_center}]$, the transformed left edge $\text{best\_center} - \text{best\_len}$ is the separator just before the first answer character, and dividing by two gives $\text{start} = (\text{best\_center} - \text{best\_len}) // 2$, with the answer being $s[\text{start} : \text{start} + \text{best\_len}]$.

```python
def manacher(s):
    """Return transformed-center radii for palindromes in s.

    The transform is conceptually #s0#s1#...#s_{n-1}#, but the implementation
    uses a private separator object so ordinary input strings may contain '#'.
    p[i] is the number of successful outward steps around transformed center i;
    the same value is the length of the original-string palindrome represented
    by that center.
    """
    if not s:
        return [0]

    sep = object()
    t = [sep]
    for ch in s:
        t.append(ch)
        t.append(sep)

    n = len(t)
    p = [0] * n
    c, r = 0, 0

    for i in range(n):
        if i < r:
            mirror = 2 * c - i
            p[i] = min(r - i, p[mirror])

        while (
            i - p[i] - 1 >= 0
            and i + p[i] + 1 < n
            and t[i - p[i] - 1] == t[i + p[i] + 1]
        ):
            p[i] += 1

        if i + p[i] > r:
            c, r = i, i + p[i]

    return p


def longest_palindrome(s):
    """Return a longest palindromic substring of s in O(n) time."""
    if not s:
        return ""

    p = manacher(s)
    best_len, best_center = 0, 0
    for i, radius in enumerate(p):
        if radius > best_len:
            best_len, best_center = radius, i

    start = (best_center - best_len) // 2
    return s[start:start + best_len]
```
