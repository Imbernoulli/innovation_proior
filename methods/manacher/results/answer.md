# Manacher's algorithm

## Problem

Given a string `s`, compute transformed-center palindrome radii in linear time and use them to return one longest palindromic substring.

## Method

Manacher's algorithm scans centers from left to right while carrying the palindrome seen so far with the farthest inclusive right boundary. If that palindrome has center `c` and right boundary `r`, then a new center `i < r` has a reflected center `mirror = 2*c - i` whose radius is already known. The reflected radius is valid at `i` only until the current boundary, so the initialization is

```python
p[i] = min(r - i, p[mirror])
```

Then ordinary outward expansion checks only the first unproved pair and anything beyond it. If the new palindrome reaches farther right, the carried center and boundary become `i` and `i + p[i]`.

Even-length palindromes are handled by a separator transform, conceptually `#s0#s1#...#s_{n-1}#`. An even palindrome in `s` becomes an odd palindrome centered on a separator; an odd palindrome in `s` remains centered on its character. The implementation below uses a private separator object rather than a literal `#`, so it remains correct when `s` itself contains `#`.

With this radius convention, `p[i]` is both the number of successful outward steps in the transformed sequence and the length of the original-string palindrome represented by center `i`. If `best_center` maximizes `p` and `best_len = p[best_center]`, then the original start index is `(best_center - best_len) // 2`.

## Code

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

## Complexity

The scan is `O(n)`: every successful expansion moves the carried right boundary to a new transformed position, and each center contributes at most one terminating failed comparison. The transformed sequence and radius array use `O(n)` space.
