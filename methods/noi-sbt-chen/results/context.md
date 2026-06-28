## Research question

Maintain a dynamic set of distinct keys under insertions while supporting the order-statistic queries: report the $k$-th smallest key, and report the rank of a given key (one plus the number of stored keys less than it). The underlying binary search tree must stay balanced, keeping its height $O(\log n)$. The deliverable is a single self-contained C++17 program that reads operations from stdin and writes query answers to stdout.

## Input-output contract

The program reads an integer `q`. Each of the next `q` lines contains an operation code and one integer argument: `I v` inserts distinct 64-bit key `v`, `S k` prints the `k`-th smallest stored key, and `R v` prints one plus the number of stored keys strictly less than `v`. It writes one line to stdout for each `S` or `R` operation, and writes nothing for `I`.

## Code framework

The scaffold is a C++17 program with `int main` as the entry point. It parses the operation stream from stdin and prints the query answers to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    vector<pair<char, long long>> operations;
    operations.reserve(q);
    for (int i = 0; i < q; ++i) {
        char op;
        long long x;
        cin >> op >> x;
        operations.push_back({op, x});
    }

    vector<long long> answers;

    // TODO:

    for (long long answer : answers) {
        cout << answer << '\n';
    }
    return 0;
}
```
