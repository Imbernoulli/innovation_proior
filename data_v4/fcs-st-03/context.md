# Weighted dictionary scoring of a text

## Research question

You are given a dictionary of `m` patterns, each a non-empty lowercase string with an integer
weight `w[i]` (weights may be negative), and one lowercase text `T`. Define the **score** of the text
as

```
score = sum over all patterns i of  w[i] * (number of occurrences of pattern i in T)
```

where occurrences are counted at **every** starting position, including overlapping ones (e.g. the
pattern `aa` occurs three times in `aaaa`). The same pattern string may appear several times in the
dictionary with different weights; each listing contributes independently. Output the total score.

This is the multi-pattern weighted text-scanning problem that underlies spam scoring, keyword
relevance ranking, and content filtering: a single text is graded against a large fixed dictionary,
and the cost that matters is processing the text once against *all* patterns at once rather than once
per pattern.

## Input / output contract

- Input (stdin):
  - The first token is `m` (`0 <= m <= 10^5`), the number of dictionary entries.
  - Then `m` lines follow, each `p_i w_i`: a non-empty pattern `p_i` over `a`..`z` and an integer
    weight `w_i` (`-10^9 <= w_i <= 10^9`).
  - Then the text `T` over `a`..`z`. The text may be empty (no token), in which case the score is `0`.
- Output (stdout): a single line with the total score as described above.
- Constraints: the sum of all pattern lengths plus the text length is at most `10^6`
  (`sum |p_i| + |T| <= 10^6`).
- Time limit: 1 second. Memory: 256 MB.

Example: for the dictionary `{ab: 5, bc: 3, ab: 2}` and text `ababcab`, `ab` occurs 3 times
(positions 0, 2, 5) and `bc` occurs once (position 3), so the score is `(5+2)*3 + 3*1 = 24`.

## Background

The objective decomposes per pattern, so two approaches are immediately on the table:

- **Per-pattern matching.** For each dictionary entry, find all its occurrences in `T` (with KMP, or
  even `string::find`) and add `w[i] * count`. Each scan is `O(|T| + |p_i|)`, so the total is
  `O(m*|T| + sum|p_i|)`. With `m` up to `10^5` and `|T|` up to `10^6`, the `m*|T|` term is the
  problem — it can be `10^11`, far beyond a one-second budget. The open question is whether the text
  can be processed **once** while simultaneously accounting for every pattern.

- **A multi-pattern automaton.** Build a single finite automaton from the whole dictionary, drive the
  text through it character by character, and read off all pattern occurrence counts from the run.
  This is the Aho-Corasick family. The open questions are how to recover *each pattern's* occurrence
  count from a single pass, and how to make the automaton's transition function total so the scan is
  genuinely `O(|T|)`.

## Evaluation settings

Judged on hidden tests covering: empty text and `m = 0`; patterns with no occurrence; a pattern equal
to the whole text; patterns longer than the text; duplicate patterns with differing (and cancelling)
weights; deeply nested patterns (`a`, `aa`, `aaa`, ... ) on a single-character text where overlapping
counts are large; zero and negative weights (so the total may be negative); and large instances with
`sum |p_i| + |T|` near `10^6` and weights near `10^9`. In the duplicate-heavy extreme — say `10^5`
copies of the single-character pattern `a` each weighted `10^9`, scored against a text of `~10^6`
copies of `a` — the total reaches `~10^20`, which overflows signed 64-bit; the intended solution must
accumulate in a 128-bit integer (and print it manually). A `long long` accumulator is a silent wrong
answer on such tests.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    if (!(cin >> m)) return 0;
    for (int i = 0; i < m; i++) {
        string p;
        long long w;
        cin >> p >> w;
        // TODO: insert pattern p with weight w into the dictionary structure.
    }

    string text;
    cin >> text;

    // TODO: score the text against all patterns at once and print the total weight
    //       of all (overlapping) occurrences.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
