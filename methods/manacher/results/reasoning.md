The thing in front of me is a string `s` of length `n`, and I want one longest substring that reads the same forwards and backwards. The obvious way to organize the problem is by centers. A palindrome either has a real character at its center, like `aba`, or it has a gap at its center, like `abba`. If I expand around every character and every gap, compare the two outward characters, and stop at the first mismatch or boundary, I get the right answer. The cost is the problem. On `s = "aaaa...a"`, almost every center expands almost all the way to the ends, so the same equalities are rediscovered again and again. There are linearly many centers and a center can spend linearly many comparisons, so the direct center expansion is quadratic.

The waste has a very specific shape. When I expand around one center and find a long palindrome, I have learned that a whole interval mirrors around that center. Then the next center inside that interval is not starting in blank space; its left neighborhood is already reflected on the right side. I should carry the palindrome that currently reaches farthest to the right, because that is the one whose information can help the largest suffix of centers I have not processed yet. Let its center be `c`, and let its right edge be `r`, inclusive, so the known palindrome reaches from `2*c - r` through `r`.

Now take a new center `i`. If `i` is outside that known interval, I cannot borrow anything and I start with radius zero. If `i` lies inside it, then its reflected center is `mirror = 2*c - i`, and I have already computed the radius there. The palindrome around `mirror` reflects into a palindrome around `i`, but only for the part that stays inside the known interval. If the reflected radius would run past `r`, the old symmetry says nothing about the character just beyond `r`; that character has not been compared yet. So the largest safe initial radius is `min(r - i, p[mirror])`. That order matters: initialize from the mirror, clamp to the current right edge, and only then run direct expansion from the first unproved pair.

The boundary convention stays consistent if `r` is inclusive. The palindrome at center `i` with radius `p[i]` spans indices `i - p[i]` through `i + p[i]`, so the center is inside the reusable interval when `i < r`. After the mirror initialization, the next attempted comparison is at `i - p[i] - 1` and `i + p[i] + 1`. If they are in bounds and equal, I increase `p[i]`; otherwise the radius is final. When this center reaches farther right than any previous center, `i + p[i] > r`, I replace the carried pair with `c = i` and `r = i + p[i]`.

This still has a while loop at every center, so I need the amortized count to be honest. Before expansion starts, the clamp guarantees `i + p[i] <= r` whenever `i` is inside the carried interval. A successful comparison can happen only when it verifies a new pair at the current right frontier or beyond it; every such success pushes the right edge of the current center one step farther, and after the update the carried `r` never moves backward. The right edge can advance only across the transformed array, so all successful comparisons together are linear. A center can also pay one final failed comparison, and there are only linearly many centers. The bookkeeping is constant per center, so the whole pass is linear.

I still have the odd/even split. Keeping two algorithms, one for character centers and one for gap centers, would duplicate the same mirror argument with slightly different indices. I can avoid that by making every original gap into a real transformed position. Conceptually I insert a separator, written as `#`, between every pair of characters and at both ends: `abba` becomes `#a#b#b#a#`. Then an even palindrome in the original string is centered on one of those separators, while an odd palindrome remains centered on its original character. Every palindrome I need to consider in the transformed sequence is now an odd-centered palindrome.

One implementation detail matters: a literal `#` is only safe if it is outside the input alphabet. Python strings can contain `#`, so the code should use a private separator object in the transformed list while the derivation still thinks of it as `#`. Real characters sit at odd transformed indices, separators at even transformed indices, and no input character can compare equal to the separator object.

With this transformed sequence, I use the radius convention from the mirror pass: `p[i]` is the number of successful outward steps around transformed center `i`. The palindrome occupies transformed indices `i - p[i]` through `i + p[i]`. Because separators and real characters alternate, the number of original characters covered by that transformed palindrome is exactly `p[i]`. For `#b#b#`, centered on the middle separator, the transformed radius is `2`, and the original palindrome length is `2`; for `#a#`, the radius is `1`, and the original length is `1`; a bare separator has radius `0`, the empty palindrome. So the longest original length is `max(p)`.

The start index comes from the same layout. Original character `s[j]` is at transformed index `2*j + 1`. If the winning transformed center is `best_center` and the original length is `best_len = p[best_center]`, then the transformed left edge is `best_center - best_len`, which is the separator position just before the first original character of the answer. Dividing by two gives `start = (best_center - best_len) // 2`, and the substring is `s[start:start + best_len]`.

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

The final shape is exactly the reuse I needed: each center first receives the radius forced by the reflected center inside the current inclusive right boundary, that radius is clamped before any expansion, direct comparisons begin only at the first unverified pair, and the right boundary only advances. The separator transform folds even and odd original palindromes into one odd-centered pass, and the transformed radius reads back directly as the original substring length.
