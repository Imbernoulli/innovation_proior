## Problem

Given a string of length $n$, maintain the number of distinct non-empty palindromic substrings while the string is read one character at a time. After each appended character, the current prefix's count should be available without rebuilding from scratch. The deliverable is a single self-contained C++ program that reads from stdin and writes to stdout. The program reads one string, prints the final number of distinct non-empty palindromic substrings, then prints a second line with the running count after each prefix.

## Code framework

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    string text;
    if (!(cin >> text)) text = "";

    long long total = 0;
    vector<long long> per_prefix(text.size(), 0);

    // TODO: fill total and per_prefix for the input.

    cout << total << "\n";
    for (size_t i = 0; i < per_prefix.size(); ++i) {
        cout << per_prefix[i] << (i + 1 < per_prefix.size() ? ' ' : '\n');
    }
    if (per_prefix.empty()) cout << "\n";
    return 0;
}
```
