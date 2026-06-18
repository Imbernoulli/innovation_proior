I need one number for every starting position in a string: how many characters from that position match the front of the string. If I write `z[i]` for that number, then `z[0]` is not useful as a measurement, so I set it to `0` by convention. The most literal computation is clear: at each `i`, compare `s[0]` with `s[i]`, then `s[1]` with `s[i + 1]`, and keep going until either the characters differ or the suffix runs out. That computes the right value, but on a string like `"aaaaa"` it compares almost the same run again and again: `z[1]` scans four characters, `z[2]` scans three, then two, then one. The sum is quadratic. The waste is not mysterious; I learn a long prefix match and then throw that information away before the next position.

So what should I keep? A nonzero value at some earlier start `l` tells me that a block beginning at `l` is an exact copy of the prefix for a certain length. If the copy reaches through index `r`, then `s[l..r]` equals `s[0..r-l]`. Among all such copied prefix blocks I have already discovered, the useful one is the one reaching farthest to the right. I will keep its endpoints as `l` and `r`. They mark the farthest certified copied region; beyond `r`, I have no information yet.

Now suppose I am computing `z[i]`. If `i` is to the right of `r`, none of the certified region touches it, so there is nothing to reuse. I start `z[i]` at zero and compare forward directly. If that direct comparison finds a match that reaches farther than the old `r`, I replace the saved region with this new one.

The interesting case is when `i` lies inside the saved region. Since `s[l..r]` is a copy of the prefix, the characters from `i` through `r` are the same as the characters from `i - l` through `r - l` near the front of the string. The index `i - l` is smaller than `i`, so its Z-value is already known. That suggests copying `z[i - l]` as the answer at `i`.

But that copy is only justified while I stay inside the certified region. If `z[i - l]` is shorter than the remaining certified length `r - i + 1`, the known mismatch for the mirrored position occurs inside the copied region, so the same mismatch occurs at `i`; the value is exactly `z[i - l]`, with no new comparison needed. If `z[i - l]` reaches the edge or goes past it, then I only know for sure that `z[i]` is at least `r - i + 1`. Past `r`, the saved region says nothing. I have to continue with direct comparisons from that boundary.

That gives the computation. For each `i`, if `i <= r`, I seed `z[i]` with `min(z[i - l], r - i + 1)`. Then I run the ordinary forward comparison from that seeded length:

```python
while i + z[i] < n and s[z[i]] == s[i + z[i]]:
    z[i] += 1
```

After the loop, if this match reaches past the current right endpoint, I set `l = i` and `r = i + z[i] - 1`.

The time bound comes from the fact that direct comparisons only matter when they move the right endpoint. Outside the region, a successful comparison reaches a position past the old `r`. At the edge case inside the region, the first unchecked comparison is also at `r + 1`. In the strict inside case, there is no loop work at all because the mismatch was already certified. The endpoint `r` only moves right and never exceeds `n - 1`, so the total number of successful direct comparisons over the whole run is linear; the failed stopping comparisons add at most one per `i`. The whole Z-array computation is `O(n)`.

For pattern matching, an occurrence of `pattern` at text index `j` means that `pattern` and `text[j:]` have a common prefix of length exactly `len(pattern)`. I can make the pattern the prefix of a larger sequence, place a unique separator after it, append the text, and compute one Z-array. The separator must be a value that cannot equal any text character, so using a fresh `object()` sentinel is cleaner than assuming some character is absent. In `combined = list(pattern) + [separator] + list(text)`, the text position `j` appears at index `j + len(pattern) + 1`. A Z-value there cannot run through the separator unless the text somehow contains that exact sentinel, which it cannot. Therefore the value is exactly `len(pattern)` precisely at occurrences.

Now I can fill the two functions:

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

The direct scans no longer repeat old work: inside the saved copied region I reuse the mirrored Z-value but clamp it to the certified length, and every direct expansion starts at the first unknown position beyond the current right endpoint. One monotone endpoint pays for all expansions, so the array is linear-time. The occurrence finder is then one more Z-array computation on the separated pattern-plus-text sequence, with index shift `len(pattern) + 1`.
