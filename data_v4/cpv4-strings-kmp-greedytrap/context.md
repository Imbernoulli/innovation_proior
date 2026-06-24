# Tiling a banner with logo prefixes

## Research question

A print shop owns a single rubber stamp engraved with a *logo word* `s`. The stamp can be inked and
pressed, but the operator may mask off the tail of the stamp and print only a **prefix** of the logo:
any one of `s[0..0]`, `s[0..1]`, ..., up to the whole word `s`. Each press lays down one such prefix,
left to right, and presses are laid end to end with no gaps and no overlaps.

You are given the logo `s` and a target banner string `t`. You must reproduce `t` *exactly* by a
sequence of presses, where every press prints some non-empty prefix of `s`, and the printed prefixes,
concatenated in order, equal `t`. Among all valid ways to do this, **minimize the number of presses**
(blocks). If `t` cannot be reproduced this way at all, report that it is impossible.

Output the minimum number of blocks, or `-1` if no valid tiling exists.

This is the kind of subproblem that hides inside text compression, run-length / dictionary coders, and
protocol framing, where a chunk must be one of a fixed family of prefixes. Getting it right means
resisting the obvious "grab the longest matching prefix and move on" reflex, which is both suboptimal
and capable of walking into a dead end.

## Input / output contract

- Input (stdin): two whitespace-separated tokens.
  - The first token is the logo `s` (`1 <= |s| <= 2*10^5`), lowercase letters `a`..`z`.
  - The second token is the banner `t` (`1 <= |t| <= 2*10^5`), lowercase letters `a`..`z`.
- Output (stdout): a single line with the minimum number of blocks, or `-1` if `t` admits no tiling
  into non-empty prefixes of `s`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `s = "babba"` and `t = "babbabb"` the answer is `2`: split `t` as `bab | babb`, both
prefixes of `babba`. (Pressing the whole logo `babba` first leaves `bb`, which then needs two more
single-`b` presses — that is `3` blocks, worse.)

## Background

A block is legal at position `i` of `t` exactly when some prefix of `s` matches `t` starting at `i`.
Two families of approach present themselves before committing:

- **Greedy by longest match.** At the current position, press the longest prefix of `s` that matches
  the banner here, advance past it, and repeat. It is `O(|t|)` after preprocessing and trivial to
  write; the open question is whether grabbing the longest legal block now is ever globally wrong.
- **Reachability dynamic programming.** For each banner position `i`, determine the set of legal block
  lengths, then compute the minimum number of blocks to reach the end. The open question is how to get
  the legal-length sets and the minimization both correct and fast enough at `|t| = 2*10^5`.

The legal-length structure is special: if a prefix of `s` of length `L` matches at `i`, then every
shorter prefix also matches at `i` (a prefix of a matching prefix). So the legal lengths at `i` form a
full interval `1..reach[i]`, where `reach[i]` is the longest prefix of `s` matching `t` at `i`. The
`reach` array is exactly what string matching (KMP / Z-function) computes.

## Evaluation settings

Judged on hidden tests covering: banners that are exact concatenations of logo prefixes (feasible,
varying block counts), banners that cannot be tiled at all (`-1`), highly self-overlapping logos such
as `aa..a` and `abab..` (where longest-match greedy is most tempting and most wrong), single-character
logos and banners, and large `|s|, |t| = 2*10^5` cases that demand a linear algorithm.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s, t;
    if (!(cin >> s >> t)) return 0;
    int m = (int)s.size();
    int n = (int)t.size();

    // TODO: compute the minimum number of non-empty prefixes-of-s whose concatenation equals t,
    //       or -1 if t cannot be tiled this way.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
