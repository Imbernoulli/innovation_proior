I first pin down the scale: `n <= 3000`, so an `O(n^2)` algorithm is acceptable, while anything like enumerating all partitions is impossible because there are `2^(n-1)` ways to cut a string. A cubic dynamic program that checks every substring by scanning it would also be too slow at about `27e9` character comparisons in the worst case. Memory-wise, `O(n^2)` booleans would fit, but I can avoid that and keep only `O(n)` state.

The natural dynamic programming state is:

`dp[i] = minimum number of palindromic pieces needed to cover s[0..i-1]`.

Then if `s[l..r]` is a palindrome, I can end the partition with that piece and update:

`dp[r + 1] = min(dp[r + 1], dp[l] + 1)`.

So the remaining task is to enumerate all palindromic substrings efficiently. I reject checking each `s[l..r]` from scratch because that makes the algorithm cubic. Instead, I expand around every possible palindrome center. For each index `c`, I expand odd-length palindromes with `(l, r) = (c, c)`, and even-length palindromes with `(l, r) = (c, c + 1)`. Every palindrome has exactly one such center, so this enumerates all palindromic substrings in total `O(n^2)` time.

I need to be careful that updating `dp` while enumerating palindromes left-to-right is valid. When processing a palindrome `s[l..r]` centered at `c`, we use `dp[l]`. Since `l <= c`, all palindromic pieces that could end before `l` have centers strictly before or at positions already processed enough to finalize `dp[l]`. More concretely, any final piece covering a prefix ending at `l` must end at `l - 1`, so its center is before `l`, and therefore before the current center. Thus `dp[l]` is already settled when it is used. This lets me avoid storing a full palindrome table.

I verify on a small example, `s = "aab"`.

Start with `dp = [0, inf, inf, inf]`.

Center `0`:
- Odd palindrome `"a"` at `[0,0]`: `dp[1] = min(inf, dp[0] + 1) = 1`.
- Even palindrome `"aa"` at `[0,1]`: `dp[2] = min(inf, dp[0] + 1) = 1`.

Center `1`:
- Odd palindrome `"a"` at `[1,1]`: `dp[2] = min(1, dp[1] + 1) = 1`.

Center `2`:
- Odd palindrome `"b"` at `[2,2]`: `dp[3] = min(inf, dp[2] + 1) = 2`.

The answer is `2`, corresponding to `"aa" | "b"`. That matches brute force by inspection: one piece is impossible because `"aab"` is not a palindrome, and two pieces works.

As another sanity check, `"abbae"` gives `2` via `"abba" | "e"`, `"abc"` gives `3`, and `"aaaa"` gives `1`. The center-expansion DP handles all of these without special cases. The final complexity is `O(n^2)` time and `O(n)` memory, which fits comfortably for `n = 3000`.