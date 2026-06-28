I need one number for every starting position in a string: how many characters from that position match the front of the string. If I write `z[i]` for that number, then `z[0]` is not useful as a measurement, so I set it to `0` by convention. The most literal computation is clear: at each `i`, compare `s[0]` with `s[i]`, then `s[1]` with `s[i + 1]`, and keep going until either the characters differ or the suffix runs out. That computes the right value, but on a string like `"aaaaa"` it compares almost the same run again and again: `z[1]` scans four characters, `z[2]` scans three, then two, then one. The sum is quadratic. The waste is not mysterious; I learn a long prefix match and then throw that information away before the next position.

So what should I keep? A nonzero value at some earlier start `l` tells me that a block beginning at `l` is an exact copy of the prefix for a certain length. If the copy reaches through index `r`, then `s[l..r]` equals `s[0..r-l]`. Among all such copied prefix blocks I have already discovered, the useful one is the one reaching farthest to the right. I will keep its endpoints as `l` and `r`. They mark the farthest certified copied region; beyond `r`, I have no information yet.

Now suppose I am computing `z[i]`. If `i` is to the right of `r`, none of the certified region touches it, so there is nothing to reuse. I start `z[i]` at zero and compare forward directly. If that direct comparison finds a match that reaches farther than the old `r`, I replace the saved region with this new one.

The case where `i` lies inside the saved region is the one that has to earn its keep. Since `s[l..r]` is a copy of the prefix, the characters from `i` through `r` are the same as the characters from `i - l` through `r - l` near the front of the string. The index `i - l` is smaller than `i`, so its Z-value is already known. That suggests copying `z[i - l]` straight into `z[i]`.

But I should be careful about how far that copy is actually licensed. The mirror equality `s[i..r] == s[i-l..r-l]` only holds out to `r`; past `r` the saved region certifies nothing. So I have to split on whether the known value `z[i - l]` fits inside the certified remainder `r - i + 1` or not. Let me reason through both halves before I trust either.

If `z[i - l]` is strictly shorter than `r - i + 1`, then the prefix-match that starts at `i - l` already mismatches at a position that is still strictly inside the mirror window. The character causing that mismatch at `i - l + z[i-l]` equals the character at `i + z[i-l]` (still `<= r`), so the same mismatch happens at `i`, and `z[i]` cannot exceed `z[i - l]`. It also cannot fall short of it, since every matched character before the mismatch is mirrored too. So in this half the answer is `z[i - l]` and no character comparison is needed. I want to be sure I am not fooling myself here, so let me hold that claim and check it against a real run below rather than declaring it settled.

If instead `z[i - l]` reaches the edge `r - i + 1` or beyond, then the mirror runs out before the known value does. Everything up to `r` is guaranteed to match — that gives `z[i] >= r - i + 1` — but the character at `r + 1` is outside the certified region, so I genuinely do not know it yet. Here I must keep comparing forward from the boundary.

Both halves are captured by seeding `z[i]` with `min(z[i - l], r - i + 1)` and then running the ordinary forward comparison from that seeded length:

```python
while i + z[i] < n and s[z[i]] == s[i + z[i]]:
    z[i] += 1
```

In the strict-inside half the seed is already the exact answer and the very first comparison fails (it is the certified mismatch), so the loop does nothing. In the edge half the seed is `r - i + 1` and the loop picks up exactly at the first unknown character. After the loop, if this match reaches past the current right endpoint, I set `l = i` and `r = i + z[i] - 1`.

Before going further I want to actually run this, because the two-case split is the kind of thing that looks right and is off by one in practice. Take `s = "aaaaa"`. At `i = 1` nothing is saved, so I compare directly: all four remaining characters match, `z[1] = 4`, and the region becomes `(l, r) = (1, 4)`. Now `i = 2` is inside `[1,4]`. The mirror is `z[i - l] = z[1] = 4` and the certified remainder is `r - i + 1 = 3`, so I seed `min(4, 3) = 3`. That is the edge half (the known value overran the window), yet the loop's first test is `s[3] == s[5]`, which is out of bounds, so it stops and `z[2] = 3`. Continuing, `i = 3` seeds `min(z[1], 2) = min(4,2) = 2`, `i = 4` seeds `min(z[1], 1) = 1`, both with no extension. The array is `[0, 4, 3, 2, 1]`, which is exactly the value the convention demands. Good — the clamp did the right thing even though every position here landed in the edge half.

The strict-inside half did not actually fire in that example, so I have not really tested the "no comparison, answer is `z[i-l]`" claim. I need a string where a mismatch sits inside the prefix. Try `s = "aaabaab"`. Direct scan at `i = 1` gives `z[1] = 2` (matches `aa`, then `s[2]='a'` vs `s[3]='b'` fails), region `(1, 2)`. At `i = 2`, inside `[1,2]`: mirror `z[i-l] = z[1] = 2`, remainder `r - i + 1 = 1`, seed `min(2,1) = 1`; first loop test `s[1]=='a'` vs `s[3]=='b'` fails, so `z[2] = 1`. At `i = 3` we are past `r`, scan directly: `s[0]='a'` vs `s[3]='b'` fails immediately, `z[3] = 0`. At `i = 4`, past `r`, scan: `aa` matches then mismatch, `z[4] = 2`, region jumps to `(4, 5)`. At `i = 5`, inside: mirror `z[5-4] = z[1] = 2`, remainder `1`, seed `1`, loop test fails, `z[5] = 1`. At `i = 6`, past `r`, `z[6] = 0`. Result `[0, 2, 1, 0, 2, 1, 0]` — matching the required output. I still want a genuine strict-inside hit, though, where the seed comes from the mirror and not from the clamp.

Take `s = "aabaabaab"`. The direct scan at `i = 3` matches `aabaab` and gives `z[3] = 6`, region `(3, 8)`. Now `i = 6` is inside `[3,8]`: mirror `z[i - l] = z[3] = 6`, remainder `r - i + 1 = 3`, seed `min(6, 3) = 3` — this is the edge half again. But `i = 4` is the strict-inside case I wanted: mirror `z[1] = 1`, remainder `r - i + 1 = 5`, seed `min(1, 5) = 1`. Here the known value `1` is strictly less than the window `5`, so my claim says `z[4]` should be exactly `1` with no comparison. Running the loop: first test is `s[1]='a'` vs `s[5]='a'`, which *matches*. So the loop would step `z[4]` to `2` — that contradicts what I claimed!

I have to stop and find the error, because the trace and my argument disagree. Let me recompute `z[4]` honestly by hand: `s = a a b a a b a a b`, suffix from index 4 is `a b a a b`, prefix is `a a b a a`. Compare: `a==a`, then `b` vs `a` — mismatch. So `z[4] = 1` after all. So the loop's first test must actually be the mismatch, not a match. Where did I go wrong? The loop compares `s[z[i]]` with `s[i + z[i]]`, i.e. with the seed `z[4] = 1` it tests `s[1]` against `s[4 + 1] = s[5]`, and `s[5] = 'b'`, not `'a'`. I misread the index: `s[5]` is the sixth character of `aabaabaab`, which is `b`. So `s[1]='a'` vs `s[5]='b'` mismatches, the loop does nothing, and `z[4] = 1`. The mistake was mine in reading off `s[5]`, not in the method. That is reassuring in two ways: the strict-inside prediction held once I indexed correctly, and the near-miss shows the argument really is doing work — the "mismatch is mirrored" claim is what forces `s[5] = s[2] = 'b'`, and `s[2]` is indeed the prefix character that broke `z[1]`. Finishing the array: `i=5` strict-inside seed `min(z[2],4)=0`; `i=6` edge seed `3`, no extension; `i=7` seed `min(z[4],2)=1`; `i=8` seed `min(z[5],1)=0`. Result `[0,1,0,6,1,0,3,1,0]`, which a direct naive scan confirms.

There is one branch I have still only seen degenerate: the edge half where, after clamping, the forward loop actually extends past `r` and moves the frontier. In `"aaaaa"` the edge cases all hit the end of the string, and in `"aabaabaab"` they happened to stop immediately. I want to see the loop genuinely add length beyond the certified region, since that is the case the linear-time argument leans on. Try `s = "aabaaa"`. Scanning: `z[1] = 1` (region `(1,1)`), `z[2] = 0`, then at `i = 3` a direct scan matches `aa` and gives `z[3] = 2`, region `(3, 4)`. Now `i = 4` is inside `[3,4]`: mirror `z[1] = 1`, remainder `r - i + 1 = 1`, seed `min(1, 1) = 1` (the edge half). The loop test is `s[1]='a'` vs `s[5]='a'` — a match — so `z[4]` steps to `2`, and the next test `s[2]='b'` vs `s[6]` is out of bounds, stopping there. So `z[4] = 2`, and because `i + z[i] - 1 = 5 > r = 4`, the frontier advances to `(4, 5)`. Here the clamped seed was the certified length `1`, and the one real comparison happened at `r + 1 = 5`, pushing `r` from `4` to `5`. A naive scan of `"aabaaa"` gives `[0, 1, 0, 2, 2, 1]`, agreeing. So the edge half does extend past the frontier exactly when it should, and the comparison it spends lands at the first uncertified position.

That last observation is also the whole cost argument, so let me make it precise instead of asserting it. Every iteration of the inner loop that *succeeds* must occur at a position `>= r + 1` at the moment it runs: when `i > r` the scan starts at `i > r`; in the edge half it starts at the seeded boundary `i + (r - i + 1) = r + 1`; in the strict-inside half the loop never makes a successful step. So each successful comparison is at the current frontier or just past it, and each such success increments `z[i]` and ultimately pushes `r` rightward by at least one. `r` starts at `0`, only increases, and never exceeds `n - 1`, so across the entire run the number of successful comparisons is at most `n - 1`. The failing comparison that stops each loop costs one per position, another `n - 1` at most. So the Z-array computation does `O(n)` character comparisons total. The `"aabaaa"` trace is consistent with this: the only successful inner steps were the four at `i=1`/`i=3`/`i=4` that each advanced the frontier, and every other position spent at most its one terminating test.

Now pattern matching. An occurrence of `pattern` at text index `j` means `pattern` and `text[j:]` have a common prefix of length exactly `len(pattern)`. That is precisely what a Z-value measures, if I can arrange for `pattern` to be the prefix of one combined sequence. I will make the pattern the prefix, place a separator after it, append the text, and compute one Z-array. The separator has to be a value that cannot equal any text character; rather than assume some character is absent, I use a fresh `object()` sentinel, which compares unequal to every real element. In `combined = list(pattern) + [separator] + list(text)`, the text position `j` appears at index `j + len(pattern) + 1`. A Z-value computed there can never run through the separator, because matching the prefix forward from a text position would, at offset `len(pattern)`, have to compare the separator against a text element, and the sentinel matches nothing. So the Z-value at `j + len(pattern) + 1` is capped at `len(pattern)`, and equals it exactly when the first `len(pattern)` characters of `text[j:]` reproduce the pattern.

I do not want to take that on faith for the overlapping case, which is where naive reasoning often drops matches. Take `text = "aaaa"`, `pattern = "aa"`. Then `combined = ['a','a', sep, 'a','a','a','a']`, length `7`, and `m + 1 = 3`, so I read `z[3], z[4], z[5]` for text positions `0, 1, 2`. Computing the Z-array of `combined`: positions `1` gives `z[1]=1` (`'a'=='a'`, then `'a'` vs sep fails); position `2` is the sep, `z[2]=0`; position `3` matches `aa` then `s[2]=sep` vs `s[5]='a'` fails, `z[3]=2`; position `4` likewise `z[4]=2`; position `5` matches one `a` then runs out, `z[5]=1`; `z[6]=0`. The positions with `z = m = 2` are indices `3` and `4`, i.e. text indices `0` and `1`. Direct check: `"aaaa"` contains `"aa"` at `0, 1, 2`. But my read only flagged `0` and `1` — text index `2` lands at combined index `5`, where `z[5] = 1` because only one `a` remains. That is correct: at text index `2` the substring is `"aa"` (positions 2,3), which *is* a match, so I am missing one. Let me recount: text index `2` is combined index `2 + 2 + 1 = 5`, and `combined[5:] = ['a','a']` — that should match `pattern = "aa"` fully and give `z[5] = 2`, not `1`. I mis-scanned. Recomputing `z[5]` on `['a','a',sep,'a','a','a','a']`: compare `s[0]='a'` vs `s[5]='a'` match, `s[1]='a'` vs `s[6]='a'` match, then `s[2]=sep` vs `s[7]` out of bounds, stop. So `z[5] = 2` after all. The flagged positions are `3, 4, 5`, i.e. text indices `0, 1, 2` — all three overlapping occurrences. The method does report overlaps; my hand-scan had again undercounted by stopping early. Running the real code on `("aaaa","aa")` returns `[0, 1, 2]`, and on `("ababab","aba")` returns `[0, 2]`, both matching brute force, so I trust the index shift `j + m + 1` and the equality test `== m`.

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

The direct scans no longer repeat old work: inside the saved copied region I reuse the mirrored Z-value but clamp it to the certified length, and every direct expansion starts at the first unknown position beyond the current right endpoint. The empty-pattern case is handled separately because there is no meaningful Z-position for it — by the usual convention every text index (and the position past the end) counts as a match, so I return `range(len(text) + 1)`. The single monotone frontier `r` pays for all expansions, so the Z-array is linear-time, and the occurrence finder is one more Z-array computation on the separated pattern-plus-text sequence with index shift `len(pattern) + 1`, giving total work proportional to `len(pattern) + len(text)`.
