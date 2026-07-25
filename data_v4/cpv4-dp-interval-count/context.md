# Counting the True-parenthesizations of a boolean chain

## Research question

You are given one boolean expression as a single token: an odd-length string that alternates a
**literal** (`T` for true, `F` for false) and a binary **operator** (`&` = and, `|` = or, `^` = xor),
starting and ending with a literal — for example `T|T&F^T`. The string carries **no parentheses**, so
its value is ambiguous until you decide a full bracketing. A *full parenthesization* is any way of
inserting brackets so the whole expression reduces to one boolean by a sequence of binary operations;
two parenthesizations are **different** if their bracket structure differs (i.e. they correspond to
different binary trees over the literals), even if they happen to evaluate the same way.

Count how many distinct full parenthesizations make the **whole expression evaluate to `T` (true)**,
and output that count **modulo `1 000 000 007`**.

## Input / output contract

- Input (stdin): a single whitespace-delimited token `s`. If `s` is well-formed it has odd length
  `1 <= |s| <= 999`, characters at even indices are in `{T, F}` and characters at odd indices are in
  `{&, |, ^}`. The input may also be **empty** (no token at all) or **malformed** (any other string);
  in those cases the answer is `0` (there is no valid expression to parenthesize).
- Output (stdout): a single line with the number of true-yielding full parenthesizations, taken
  modulo `1 000 000 007`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = T|T&F^T` the answer is `4`.

## Evaluation settings

Judged on hidden tests covering: the three operators in isolation and mixed; single-literal inputs
(`T`, `F`); strings where the true-count is `0` and where it equals the total number of
parenthesizations; the empty and malformed inputs (answer `0`); cases that stress the modulus; and
maximum-length well-formed strings (`|s| = 999`, i.e. `500` literals) to check the solution finishes
inside the time limit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) { cout << 0 << "\n"; return 0; }   // empty input -> 0

    const long long MOD = 1000000007LL;

    // Parse s into literals val[] (1=T,0=F) and operators op[]; reject malformed -> 0.
    // ...

    // TODO: count the full parenthesizations of s that evaluate to true, modulo MOD.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
