# Reconstruct a string from its prefix function (border profile)

## Research question

You are handed the **prefix-function array** `pi[0..n-1]` of some unknown string and asked to recover a
string that produces it — or to certify that no string can. Concretely, for a string `s` of length `n`
over some alphabet, the prefix function is

```
pi[i] = length of the longest proper prefix of s[0..i] that is also a suffix of s[0..i].
```

(`pi[i]` is the length of the longest *border* of the prefix `s[0..i]`; "proper" means shorter than the
prefix itself.) Given only the integer array `pi`, output **any** string over the lowercase Latin
alphabet whose prefix function equals `pi` exactly, or output `-1` if the array is not the prefix
function of any string.

The subtlety is that most integer arrays are **not** valid prefix functions. The values are heavily
constrained by how borders of consecutive prefixes relate to one another, and those constraints compose
**transitively** along the border chain — a local "looks legal" check is not enough. Recognising and
testing that transitive feasibility is the heart of the problem.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `pi[0], ..., pi[n-1]`,
  whitespace-separated. Each `pi[i]` is given as an integer (it *should* satisfy `0 <= pi[i] <= i`, but
  the input may contain out-of-range or otherwise invalid values that you must reject).
- Output (stdout):
  - if some string has exactly this prefix function, print one such string on a single line, using only
    characters `a`–`z` (any valid string is accepted);
  - otherwise print `-1`.
  - For `n = 0` the empty string is the unique answer; print an empty line.
- Time limit: 1 second. Memory: 256 MB.

Example. For `pi = [0, 0, 1, 2, 3, 0]` a valid answer is `ababac` (its prefix function is exactly
`[0,0,1,2,3,0]`). For `pi = [0, 1, 1]` the answer is `-1`: a string with `pi[1]=1` must look like `xx?`,
and then `pi[2]` is forced to be either `2` (if `?=x`) or `0` (otherwise) — never `1`.

## Background

The prefix function is the table that powers KMP string matching. The recurrence used to *compute* it
from a known string is standard:

```
pi[0] = 0;
for i = 1..n-1:
    k = pi[i-1];
    while (k > 0 && s[i] != s[k]) k = pi[k-1];
    if (s[i] == s[k]) k = k + 1;
    pi[i] = k;
```

This problem is the **inverse**: the string is hidden and only the table is given. Two facts about the
table are easy to see and necessary:

- `pi[0] = 0` always, and `0 <= pi[i] <= i`;
- a border can grow by at most one per character, so `pi[i] <= pi[i-1] + 1`.

These conditions are necessary but **not sufficient**. `[0,1,1]` satisfies all of them yet is
infeasible. The missing ingredient is that the characters forced along the *border chain*
`pi[i-1] -> pi[pi[i-1]-1] -> ... -> 0` must be mutually consistent. The two candidate routes a solver
weighs are (a) check feasibility with a clever closed-form rule over the array, versus (b) actually
build a witness string and confirm it.

## Evaluation settings

Judged on hidden tests covering: genuine prefix functions of random small- and large-alphabet strings;
near-miss arrays formed by perturbing one entry of a real prefix function (the dangerous `-1` cases);
fully random integer arrays; structural edges (`n = 0`, `n = 1` with `pi=[0]` and the invalid `pi=[1]`,
constant arrays, strictly increasing arrays); deep border-chain inputs (Zimin / ruler words) where the
chain length is `Theta(log n)`; and large `n = 2*10^5`. Any string with the correct prefix function is
accepted, so outputs are checked by recomputing the prefix function, not by string equality.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;          // empty input -> n = 0
    vector<int> pi(n);
    for (int i = 0; i < n; i++) cin >> pi[i];

    // TODO: decide whether pi is a realizable prefix function; if so build a
    // witness string over 'a'..'z', otherwise report -1.
    string answer = "-1";

    cout << answer << "\n";
    return 0;
}
```
