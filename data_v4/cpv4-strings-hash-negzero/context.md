# Longest repeated block in an integer sequence

## Research question

You are given a sequence of `n` integers `a[0..n-1]` whose values may be **negative, zero, or
positive**. A *block* is a contiguous subarray, identified by its sequence of values (two blocks are
equal exactly when they have the same length and the same values in the same order). A block is
**repeated** if it occurs at two or more *distinct* starting positions; the two occurrences are
allowed to overlap. Output the length of the longest repeated block, or `0` if no block of length
`>= 1` repeats.

This is the integer-alphabet analogue of "longest repeated substring." It shows up whenever you must
detect duplicated runs in numeric data — repeated motifs in a signal, the longest period that recurs
in a log of deltas, or the longest tandem-friendly segment in a difference array. Getting the
numeric-alphabet version exactly right — including the all-negative, all-zero, and empty-array
corners — is the whole point: a hash that is fine for lowercase letters silently breaks once the
"characters" can be `0` or negative.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated (newlines or spaces, any layout).
- Output (stdout): a single line with the length of the longest repeated block, or `0` if none.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `a = [4, -1, 0, 0, 4, -1, 0, 7]` the answer is `3`: the block `[4, -1, 0]` starts at
index `0` and again at index `4`. No block of length `4` repeats, so `3` is the longest.

## Background

Two structural facts drive every correct approach.

- **Monotonicity in length.** If some block of length `L` occurs at two distinct starts `i != j`,
  then its length-`(L-1)` prefix also occurs at the two distinct starts `i` and `j`. So
  "a repeated block of length `L` exists" is a monotone predicate in `L`: true for all lengths up to
  some threshold and false above it. That lets us **binary-search** the answer and only ask a yes/no
  question `hasDup(L)` for `O(log n)` values of `L`.
- **Equality of blocks via fingerprints.** Inside one `hasDup(L)` query we must detect whether any two
  of the `n - L + 1` length-`L` windows are equal. Comparing windows pairwise is `O(n^2 L)`. A
  **polynomial rolling hash** assigns each window a fingerprint computable in `O(1)` from prefix
  hashes, so a single linear scan with a hash table answers the query — *provided the hash treats the
  numeric alphabet correctly.*

The delicate part is exactly the alphabet. A textbook polynomial hash over a string treats each
character as a positive "digit" `c >= 1` in some base `B`, so the prefix recurrence
`H_i = H_{i-1} * B + c_i` is injective on equal-length strings. Two things break that here:

- a value of `0` would make a leading-zero block fingerprint-indistinguishable from the same block
  without the leading zero in some naive packings, and more importantly `0` is a perfectly legal
  symbol that must be hashable;
- a **negative** value cannot be fed into a modular hash directly — `(-3) mod p` and the modular
  arithmetic must agree, and a sign that flips a "digit" past the base destroys injectivity.

The standard cure is to **offset** every value into a strictly positive range before hashing. With
`|a[i]| <= 10^9`, mapping `v -> v + 10^9 + 1` lands every symbol in `[1, 2*10^9 + 1]`, strictly
positive, so `0` and negatives become ordinary positive digits and no symbol ever fingerprints to the
additive identity. The empty array, the all-negative array (no repeat), and the all-zero array (long
trivial repeat) are the corners that punish a wrong base case or a wrong offset.

## Evaluation settings

Judged on hidden tests covering: the empty array (`n = 0`) and `n = 1` (answer `0`); all-negative
arrays both with and without a repeat; all-zero arrays (the answer is `n-1`, an overlapping block of
zeros); tiny alphabets `{-1, 0, 1}` that force long repeats; large-magnitude values near `+-10^9`
that stress the sign offset; and large `n = 2*10^5` random and highly repetitive inputs. The count of
windows times the binary-search factor must stay inside the time limit, and the answer length can be
close to `n` (so no 32-bit assumption is needed for the value, but indices and powers must use wide
enough arithmetic for the modular hash).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) { cout << 0 << "\n"; return 0; }
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: return the length of the longest block (contiguous subarray) that occurs at two or more
    // distinct starting positions (overlaps allowed); 0 if no block of length >= 1 repeats.
    // Watch the alphabet: a[i] may be negative or zero, and n may be 0.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
