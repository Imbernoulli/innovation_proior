## Problem

Given a string $S$ of length $n$, build its suffix array: the permutation `sa` that lists all suffix starting positions in lexicographic order. Also build the inverse rank array, where `rank[sa[i]] = i`, and the adjacent longest-common-prefix array `height`, where `height[0] = 0` and `height[i]` is the LCP of the two suffixes beginning at `sa[i - 1]` and `sa[i]`.

The construction should avoid materializing all suffix strings and should run in $O(n \log n)$ time for the suffix array, plus $O(n)$ time for `height`.

## Code framework

The input sequence for the suffix-array builder has already been converted to integer codes: `r` has length `n`, every entry is in `[0, m)`, and `r[n - 1]` is a unique smallest sentinel. A stable bounded-key counting primitive is available, along with linear-size integer work arrays.

```python
def build_suffix_array(r, n, m):
    """sa[0..n-1] sorting the n suffixes of r (integer codes in [0, m),
    r[n-1] the unique smallest sentinel) lexicographically. O(n log n)."""
    sa = [0] * n
    x = list(r)            # working integer key per suffix; seed with the code
    y = [0] * n            # scratch
    ws = [0] * max(m, n)   # bounded-key tally
    # TODO
    return sa


def build_height(s, sa):
    """height[i] = LCP(Suffix(sa[i-1]), Suffix(sa[i])); height[0] = 0. O(n)."""
    n = len(s)
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i    # inverse of sa
    height = [0] * n
    # TODO
    return height
```
