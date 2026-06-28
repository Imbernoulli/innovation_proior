# Context

## Problem

Given a string `s` of length `n`, write a single self-contained C++ program that finds a longest substring that reads the same forwards and backwards.

## Research question

Produce a single self-contained C++17 program that reads from standard input and writes the required answer to standard output.

## Input-output contract

The program reads one line `s` from stdin. It prints two lines to stdout: first the length of a longest palindromic substring of `s`, then one such substring. If `s` is empty, the first output line is `0` and the second output line is empty.

## Code framework

The scaffold is a C++17 program with `main` as the entry point. It reads the input string from stdin and prints the answer to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    getline(cin, s);
    if (!s.empty() && s.back() == '\r') s.pop_back();

    string answer;

    // TODO:

    cout << answer.size() << '\n';
    cout << answer << '\n';
    return 0;
}
```
