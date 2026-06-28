# Context

## Problem

Given a string $s$ of length $n$, count the number of distinct non-empty substrings of $s$, supporting the string being built one character at a time (online), in $O(n)$ over a fixed alphabet.

## Code framework

The deliverable is a single self-contained C++17 program. It reads one whitespace-delimited string `s` from stdin and writes to stdout: first the number of distinct non-empty substrings of `s`, then the running count after each prefix `s[0..i]` as space-separated values on the second line. The string should still be handled online: assign stable dense integer codes as characters first appear, feed one code at a time into incremental state, and maintain the running answer.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {
        cout << 0 << '\n';
        return 0;
    }

    vector<long long> per_prefix;
    per_prefix.reserve(s.size());
    long long total = 0;

    unordered_map<unsigned char, int> code;
    int next_code = 0;
    for (unsigned char ch : s) {
        if (!code.count(ch)) code[ch] = next_code++;
        int c = code[ch];
        (void)c;

        // TODO: extend the online state by c and update total.
        per_prefix.push_back(total);
    }

    cout << total << '\n';
    for (size_t i = 0; i < per_prefix.size(); ++i) {
        if (i) cout << ' ';
        cout << per_prefix[i];
    }
    cout << '\n';
    return 0;
}
```
