The thing in front of me is a string `s` of length `n`, and I want one longest substring that reads the same forwards and backwards. The obvious way to organize the problem is by centers. A palindrome either has a real character at its center, like `aba`, or it has a gap at its center, like `abba`. If I expand around every character and every gap, compare the two outward characters, and stop at the first mismatch or boundary, I cover every palindrome exactly once. The cost is the problem. On `s = "aaaa...a"`, almost every center expands almost all the way to the ends, so the same equalities are rediscovered again and again. There are linearly many centers and a center can spend linearly many comparisons, so the direct center expansion is quadratic.

I want to see where the waste actually is, concretely. Take `aaaa` and expand around the center between the two middle `a`s: I verify `a==a`, then `aa==aa`, walking outward across the whole string. Now move to the next gap to its right. Direct expansion would re-verify `a==a` there from scratch — but I already know the string is symmetric about the previous center, and this new gap sits inside the palindrome I just found, so its left side is a mirror image of a region whose right side I have also already checked. The equalities I am about to redo are reflections of equalities I already proved. That is the redundancy to kill.

So I should carry, as I scan, the palindrome that currently reaches farthest to the right, because that is the one whose interior overlaps the largest suffix of not-yet-processed centers. Let its center be `c` and its right edge be `r`, inclusive, so the known palindrome reaches from `2*c - r` through `r`.

Take a new center `i`. If `i` is outside that known interval, I have nothing to borrow and I start with radius zero. If `i` lies inside it, its reflected center is `mirror = 2*c - i`, and I have already computed the radius there. The palindrome around `mirror` reflects into a palindrome around `i` — but the reflection is only trustworthy for the part that stays inside the known interval. If `mirror`'s radius would reach past `r` when copied to `i`, then it is claiming something about a character just beyond `r`, and that character has never been compared against its counterpart. The old symmetry says nothing there. So the largest radius I can assert for free is `min(r - i, p[mirror])`: take what the mirror knows, but clamp it at the current right edge. After that I run direct expansion, and crucially it begins at the first pair the clamp did not already certify.

Let me make sure the boundary convention is consistent before trusting any of this. With `r` inclusive, the palindrome at center `i` with radius `p[i]` spans `i - p[i]` through `i + p[i]`, so `i` is strictly inside the reusable interval exactly when `i < r`. After the mirror initialization sets some radius, the next pair to attempt is at `i - p[i] - 1` and `i + p[i] + 1`. If those are in bounds and equal, `p[i]` grows; otherwise it is final. And when this center reaches farther right than anything before, `i + p[i] > r`, I move the carried pair to `c = i`, `r = i + p[i]`.

I should not take the amortized claim on faith, since there is still a `while` loop at every center. Let me count comparisons on `aaaa` and see whether the total stays linear. Build the centers as the alternating layout `#a#a#a#a#` (I will justify that transform in a moment) and scan:

- center 0 (`#`): radius 0, no successful step.
- center 1 (`a`): expands once, `p=1`, the new right edge is 2. One success.
- center 2 (`#`): mirror is uninformative here (`r=2`, `i=2`, not strictly inside); expands twice, `p=2`, right edge 4. Two successes.
- center 3 (`a`): `c,r = 2,4`, `mirror = 2*2-3 = 1`, `p[1]=1`, clamp `min(4-3, 1) = 1`. So it starts at radius 1 *without* re-checking the inner pair, then expands twice more, `p=3`, right edge 6. Two successes, and one pair re-used for free.
- center 4 (`#`): `c,r = 3,6`, `mirror = 2`, `p[2]=2`, clamp `min(6-4, 2) = 2`. Starts at 2, expands twice more, `p=4`, right edge 8. Two successes.
- center 5 (`a`): `c,r = 4,8`, `mirror = 3`, `p[3]=3`, clamp `min(8-5, 3) = 3`. Starts at 3, and now the next pair is out of bounds, so zero further successes. The mirror gave the whole radius.
- centers 6, 7, 8: clamps give `2, 1, 0` from the mirror, and each does zero further successful steps.

Adding the successes: `0+1+2+2+2+0+0+0+0 = 7`. The transformed array has length 9, so 7 successful comparisons is below `n_t`. And the pattern is exactly what I hoped: once the carried right edge reached 8 at center 4, every later center got its entire radius from the clamp and spent no expansion at all. I can now see why this is general. A successful comparison certifies a pair at the current right frontier or beyond — because the clamp guarantees `i + p[i] <= r` before expansion starts, so the first pair expansion touches is at index `i + p[i] + 1 > r` only when it is genuinely new. Each such success pushes the carried right edge one step farther, and `r` never retreats. The right edge can advance at most across the whole transformed array, so all successful comparisons summed over all centers is linear. Each center also pays at most one failed comparison to stop, and there are linearly many centers. Constant bookkeeping per center, so the whole pass is linear. To be more confident this isn't an artifact of one input, I ran the scan on twenty thousand random strings and on `'a'*L` for several `L`; the count of successful steps stayed at or below the transformed length in every case (on `'a'*L` it was exactly `2L-1`, i.e. `n_t - 2`), which is the bound the argument predicts.

Now the odd/even split I waved at. Keeping two algorithms — one for character centers, one for gap centers — would duplicate the same mirror argument with two sets of off-by-one indices. I would rather have a single odd-centered pass. The trick is to make every original gap into an actual position. Insert a separator between every pair of characters and at both ends, so `abba` becomes `#a#b#b#a#`. An even palindrome in the original is now centered on a separator, an odd palindrome stays centered on its character, and every palindrome I care about is odd-centered in the transformed sequence.

Before I commit to `#` as the separator, I should check the one thing that could quietly break it: what if `#` is a real input character? If `s = "a#a"`, the transform is `#a###a#`, and now a real `#` from the input is indistinguishable from a structural separator. Expanding around the wrong middle would compare a real `#` against a structural `#` and wrongly extend. So a literal character is not safe — the separator has to be a token that no input character can equal. In Python the clean way is a sentinel `object()`; I keep `t` as a list, put the sentinel at the even positions, and real characters at the odd positions. The derivation can keep thinking of it as `#`, but the code must use the sentinel so that an input containing `#` still works. (I confirmed this: with the sentinel, `longest_palindrome("a#a")` returns `"a#a"`, matching brute force; with a literal `#` it would not.)

With the transformed sequence, `p[i]` is the number of successful outward steps around transformed center `i`, so the palindrome occupies transformed indices `i - p[i]` through `i + p[i]`. Because separators and real characters strictly alternate, the count of original characters inside that span equals `p[i]`. Let me check that the radius really reads back as an original length rather than needing a conversion. For `#b#b#` centered on the middle `#`, expanding gives `b==b` then `#==#`, radius 2, and the original palindrome is `bb`, length 2 — matches. For `#a#`, radius 1, original `a`, length 1 — matches. A bare separator has radius 0, the empty palindrome — matches. So the longest original length is just `max(p)`.

The start index comes from the same alternating layout. Original character `s[j]` sits at transformed index `2*j + 1`. If the winning transformed center is `best_center` with original length `best_len = p[best_center]`, the transformed left edge of the answer is `best_center - best_len`. That index is the separator immediately before the first original character, an even number, and halving it lands on the original index: `start = (best_center - best_len) // 2`.

Let me run the whole pipeline on a string where the answer is even-length and not at the start, so both the separator transform and the start formula get exercised non-trivially. Take `cbbd`, transform `#c#b#b#d#` (length 9). Scanning: center 1 (`c`) gets `p=1`; center 3 (`b`) gets `p=1`; center 4 (the middle `#`) expands `b==b` then `#==#`, `p=2`, the longest. Center 5 (`b`) has `c,r = 4,6`, `mirror = 3`, `p[3]=1`, clamp `min(6-5,1)=1`, no further steps — again the mirror does the work. The radius array is `[0,1,0,1,2,1,0,1,0]`, so `max(p)=2` at `best_center = 4`. Then `start = (4 - 2)//2 = 1`, and `s[1:1+2] = "bb"`. That is the correct answer, an even palindrome sitting in the interior, recovered with the right start. I also checked the answers against brute force over the twenty thousand random strings above and found no length mismatch.

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

Each center first receives the radius forced by its reflected center, clamped to the current inclusive right boundary, so direct comparisons begin only at the first unverified pair and the boundary only advances. The separator transform folds even and odd original palindromes into one odd-centered pass, and the transformed radius reads back directly as the original substring length — which the `cbbd` and `aaaa` traces above bear out end to end.
