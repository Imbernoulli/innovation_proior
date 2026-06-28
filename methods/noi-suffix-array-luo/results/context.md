## Problem

Given a string $S$ of length $n$, build its suffix array: the permutation `sa` that lists all suffix starting positions in lexicographic order. Also build the inverse rank array, where `rank[sa[i]] = i`, and the adjacent longest-common-prefix array `height`, where `height[0] = 0` and `height[i]` is the LCP of the two suffixes beginning at `sa[i - 1]` and `sa[i]`.

The construction should avoid materializing all suffix strings and should run in $O(n \log n)$ time for the suffix array, plus $O(n)$ time for `height`.

## Code framework

The deliverable is a single self-contained C++17 program. It reads one line, the string $S$, from standard input; appends the unique smallest sentinel internally; and writes `n`, then the `sa[]` row, then the `height[]` row to standard output.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    string text;
    getline(cin, text);

    vector<int> s;
    s.reserve(text.size() + 1);
    for (unsigned char c : text) s.push_back((int)c + 1);
    s.push_back(0);
    int n = (int)s.size();

    vector<int> sa(n), height(n);

    // TODO:

    cout << n << "\n";
    for (int i = 0; i < n; i++) cout << sa[i] << " \n"[i == n - 1];
    for (int i = 0; i < n; i++) cout << height[i] << " \n"[i == n - 1];
    return 0;
}
```
