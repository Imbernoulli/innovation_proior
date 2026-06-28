# Context

## Research question

Given a string `s` of length `n`, compute the Z-array, where `z[i]` is the length of the longest substring starting at `i` that is also a prefix of `s`; and use it to find all occurrences of a pattern `P` in a text `T`, all in `O(n)`. The deliverable is a single self-contained C++17 program that reads from stdin and writes to stdout.

By convention, `z[0]` is `0`. For example, the Z-array for `"aaaaa"` is `[0, 4, 3, 2, 1]`, for `"aaabaab"` is `[0, 2, 1, 0, 2, 1, 0]`, and for `"abacaba"` is `[0, 0, 1, 0, 3, 0, 1]`.

The direct approach computes each `z[i]` independently by comparing `s[0]` with `s[i]`, then `s[1]` with `s[i + 1]`, and so on until a mismatch or the end of the string. The goal is a linear-time computation of the Z-array, and then a linear-time occurrence finder whose total work is proportional to `len(pattern) + len(text)`.

## Input-output contract

The program reads the text string `s` on the first line of stdin and the pattern string `p` on the second line. It prints the Z-array of `s` as space-separated integers on the first output line, then prints the sorted start indices in `s` where `p` occurs, including overlapping occurrences, as space-separated integers on the second output line.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    string s, p;
    getline(cin, s);
    getline(cin, p);

    vector<int> z;
    vector<int> occ;

    // TODO:

    for (size_t i = 0; i < z.size(); ++i) cout << (i ? " " : "") << z[i];
    cout << "\n";

    for (size_t i = 0; i < occ.size(); ++i) cout << (i ? " " : "") << occ[i];
    cout << "\n";
    return 0;
}
```
