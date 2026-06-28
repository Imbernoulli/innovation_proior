# Manacher's algorithm

## Problem

Given a string `s`, compute transformed-center palindrome radii in linear time and use them to return one longest palindromic substring.

## Method

Manacher's algorithm scans centers from left to right while carrying the palindrome seen so far with the farthest inclusive right boundary. If that palindrome has center `c` and right boundary `r`, then a new center `i < r` has a reflected center `mirror = 2*c - i` whose radius is already known. The reflected radius is valid at `i` only until the current boundary, so the initialization is

```cpp
p[i] = min(r - i, p[mirror]);
```

Then ordinary outward expansion checks only the first unproved pair and anything beyond it. If the new palindrome reaches farther right, the carried center and boundary become `i` and `i + p[i]`.

Even-length palindromes are handled by a separator transform, conceptually `#s0#s1#...#s_{n-1}#`. An even palindrome in `s` becomes an odd palindrome centered on a separator; an odd palindrome in `s` remains centered on its character. The implementation below stores no literal separator: even transformed indices are separators and odd indices carry real characters, with the comparison treating two separators as equal and a separator as unequal to any real character, so it remains correct when `s` itself contains `#`.

With this radius convention, `p[i]` is both the number of successful outward steps in the transformed sequence and the length of the original-string palindrome represented by center `i`. If `best_center` maximizes `p` and `best_len = p[best_center]`, then the original start index is `(best_center - best_len) // 2`.

## Algorithm

Single-file C++17. It reads one line `s` from stdin (which may itself contain
`#` or any other characters) and prints the length of a longest palindromic
substring on the first line and the substring on the second. No literal
separator character is stored: in the transformed coordinate even indices are
separators and odd indices carry real characters, so the `#s0#s1#...#s_{n-1}#`
transform stays correct even when `s` contains `#`.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Manacher's algorithm. Conceptually the input s = s0 s1 ... s_{n-1} is woven
// with a separator into #s0#s1#...#s_{n-1}#, but no literal separator character
// is used: in the transformed coordinate even indices are separators and odd
// indices carry real characters, so two transformed positions are "equal" iff
// both are separators or both hold the same real character. p[i] is the number
// of successful outward steps around transformed center i, which is exactly the
// length of the original-string palindrome represented by that center.
vector<int> manacher(const string& s) {
    int n = (int)s.size();
    if (n == 0) return vector<int>(1, 0);

    int m = 2 * n + 1;                 // transformed length: separators + chars
    // same(i, j): do transformed positions i and j compare equal?
    auto same = [&](int i, int j) -> bool {
        bool sep_i = (i % 2 == 0);     // even index => separator
        bool sep_j = (j % 2 == 0);
        if (sep_i || sep_j) return sep_i && sep_j;   // separators only match separators
        return s[i / 2] == s[j / 2];                 // both real characters
    };

    vector<int> p(m, 0);
    int c = 0, r = 0;                  // carried palindrome: center c, inclusive right edge r
    for (int i = 0; i < m; ++i) {
        if (i < r) {
            int mirror = 2 * c - i;
            p[i] = min(r - i, p[mirror]);
        }
        while (i - p[i] - 1 >= 0 && i + p[i] + 1 < m &&
               same(i - p[i] - 1, i + p[i] + 1)) {
            ++p[i];
        }
        if (i + p[i] > r) {
            c = i;
            r = i + p[i];
        }
    }
    return p;
}

// Return one longest palindromic substring of s in O(n) time.
string longest_palindrome(const string& s) {
    if (s.empty()) return "";
    vector<int> p = manacher(s);
    int best_len = 0, best_center = 0;
    for (int i = 0; i < (int)p.size(); ++i) {
        if (p[i] > best_len) {
            best_len = p[i];
            best_center = i;
        }
    }
    int start = (best_center - best_len) / 2;
    return s.substr(start, best_len);
}

// Reads one line s (which may contain '#' or any other characters) from stdin
// and prints the length of a longest palindromic substring, then the substring.
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    getline(cin, s);
    if (!s.empty() && s.back() == '\r') s.pop_back();   // tolerate CRLF input

    string ans = longest_palindrome(s);
    cout << ans.size() << '\n';
    cout << ans << '\n';
    return 0;
}
```

## Complexity

The scan is `O(n)`: every successful expansion moves the carried right boundary to a new transformed position, and each center contributes at most one terminating failed comparison. The transformed sequence and radius array use `O(n)` space.
