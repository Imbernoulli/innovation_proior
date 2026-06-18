# Z-Algorithm and Linear-Time Pattern Matching

The Z-array of a sequence `s` stores, at each position `i`, the length of the longest prefix of `s` that also starts at `i` (the longest common prefix of `s` and the suffix `s[i:]`), with `z[0] = 0` by convention. Examples: `z("aaaaa") = [0, 4, 3, 2, 1]`, `z("aaabaab") = [0, 2, 1, 0, 2, 1, 0]`, `z("abacaba") = [0, 0, 1, 0, 3, 0, 1]`.

A nonzero `z[l]` certifies that `s[l : l + z[l]]` is a verbatim copy of the prefix. The linear-time computation keeps the copy `[l, r]` (so `s[l..r] == s[0..r-l]`) whose right endpoint `r` reaches farthest right; `r` is a frontier past which nothing is yet known. For a new position `i <= r`, the copy aligns `s[i..r]` with `s[i-l .. r-l]` near the front, so the earlier value `z[i-l]` transfers - clamped to the number of characters still certified, `r - i + 1`. Hence `z[i] = min(z[i-l], r-i+1)`. If `z[i-l] < r-i+1`, the mirrored mismatch lies inside the copy and `z[i] = z[i-l]` exactly, with no comparisons. Otherwise the match is only guaranteed to `r`, and direct character comparison extends it past the frontier. After each position, if the match reaches beyond `r`, the frontier moves: `l, r = i, i + z[i] - 1`.

**Amortized `O(n)`.** Every direct comparison that succeeds occurs at a position beyond the current `r` and therefore pushes `r` rightward: when `i > r` the scan starts at `i`; when `i <= r` with `z[i-l] >= r-i+1` the first unchecked comparison is at `r + 1`; and when `z[i-l] < r-i+1` no comparison runs at all (the mismatch is already certified - were it to match, `z[i-l]` would have been undercounted). The frontier `r` only increases and never exceeds `n - 1`, so the total number of successful comparisons over the whole run is at most `n`; adding one terminating comparison per `i` keeps the entire Z-array computation `O(n)`.

For pattern search, an occurrence of `pattern` at text index `j` means `pattern` and `text[j:]` share a prefix of length exactly `len(pattern)` - what the Z-array reports when `pattern` is the prefix of a longer sequence. Build `list(pattern) + [separator] + list(text)` with `separator` a fresh sentinel object that cannot equal any text element, so a prefix match starting in the text region stops at the separator after at most `len(pattern)` characters. Text index `j` lands at combined index `j + len(pattern) + 1`; a Z-value equal to `len(pattern)` there is exactly one occurrence. This reports overlapping matches and runs in `O(len(pattern) + len(text))`.

```python
def z_function(s):
    """Return the Z-array of s: z[i] is the length of the longest prefix of s
    that also starts at position i. z[0] is 0 by convention."""
    n = len(s)
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(z[i - l], r - i + 1)
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    return z


def find_occurrences(text, pattern):
    """Return the sorted list of start indices i in text where pattern occurs."""
    m = len(pattern)
    if m == 0:
        return list(range(len(text) + 1))
    separator = object()
    combined = list(pattern) + [separator] + list(text)
    z = z_function(combined)
    occurrences = []
    for j in range(len(text)):
        if z[j + m + 1] == m:
            occurrences.append(j)
    return occurrences
```
