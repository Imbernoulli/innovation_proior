# Word break: is a string a concatenation of dictionary words?

## Research question

You are given a dictionary of `n` words and a target string `s`, all over the lowercase
alphabet `a..z`. Decide whether `s` can be written as a concatenation of **one or more**
dictionary words, where **each word may be reused any number of times** and words are used
back-to-back with no gaps and no overlaps. The empty string is considered segmentable (it is
the empty concatenation). Output `YES` if such a segmentation exists, otherwise `NO`.

This is the classic *word break* decision problem.

## Input / output contract

- Input (stdin):
  - the first token is `n` — the number of dictionary words (`0 <= n <= 10^5`);
  - then `n` tokens, the dictionary words `w_1 ... w_n`, each a non-empty lowercase string;
  - then one token `s`, the string to segment. If `s` is empty it contributes no token, so
    after the `n` words there may be no further token — that case denotes the empty `s`.
- Constraints:
  - each word has length `>= 1`; the **total length of all dictionary words is `<= 2*10^5`**;
  - `s` has length `0 <= |s| <= 5000`, lowercase letters only;
  - words in the dictionary are not guaranteed distinct.
- Output (stdout): a single line, `YES` or `NO`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with dictionary `{le, leet, code, etcode, leetcode}` and `s = "leetcode"`, the answer
is `YES` (for instance `leet | code`, or `le | etcode`, or the single word `leetcode`).

## Evaluation settings

Judged on hidden tests covering: a range of adversarially constructed dictionary/string pairs;
strings built from a small alphabet where many partial matches are possible; strings containing
letters that cannot be covered by any dictionary word; the empty string and the empty dictionary;
single exact-match cases; and large inputs with `|s| = 5000` and total dictionary length near
`2*10^5`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<string> dict(n);
    for (auto &w : dict) cin >> w;
    string s;
    cin >> s;                 // empty s -> no token -> s stays ""
    int m = (int)s.size();

    // TODO: decide whether s is a concatenation of dictionary words (reuse allowed,
    // empty s is segmentable). Print "YES" or "NO".
    bool ok = false;

    cout << (ok ? "YES" : "NO") << "\n";
    return 0;
}
```
