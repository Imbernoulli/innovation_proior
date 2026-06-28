# DistinctPalindromes

## Problem

Maintain the number of distinct non-empty palindromic substrings of a string while reading it online, one character at a time.

## Method

A length-$n$ string has at most $n$ distinct non-empty palindromic substrings. When a character is appended, every new palindrome must end at the new last position, and every such end-palindrome except the longest one already has an earlier mirrored occurrence inside the old prefix. Therefore one append creates at most one new distinct palindrome: the new longest palindromic suffix.

The structure stores one node per distinct palindrome, plus two roots:

- A length-$0$ root for the empty palindrome.
- A length-$-1$ imaginary root so wrapping a character around it creates a length-$1$ palindrome.
- A transition `go[x][c]` from an inner palindrome node `x` to the palindrome formed by wrapping character `c` on both ends.
- A suffix link `fail[x]` to the longest proper palindromic suffix of node `x`.

To append character `c`, walk suffix links from the current longest palindromic suffix `last` until the candidate palindrome can be wrapped by `c`, i.e. until `s[i - 1 - length[x]] == s[i]`. The landing node `cur` gives the new longest palindromic suffix through `go[cur][c]`. If that transition is missing, create exactly one new node and set its suffix link by repeating the same search from `fail[cur]`; a single-character node links to the length-$0$ root.

The first suffix-link search over all appends is amortized linear: if the walk stops at length $L$, the new longest palindromic suffix has length $L+2$, so the failed steps are at most $\Phi_{\text{old}}-\Phi_{\text{new}}+2$ for $\Phi=$ current longest-suffix length. The second search, used only for new nodes, is the same proper-suffix-chain search and is charged by the same length-drop potential, adding only a constant multiple of the suffix-link work. Thus the total number of suffix-link jumps is $O(n)$. With array transitions over a fixed alphabet, construction is $O(n)$ time; with balanced maps it is $O(n\log\Sigma)$ time and $O(n)$ sparse transition space.

## Code

The program reads one string `s` from stdin; it prints the number of distinct non-empty palindromic substrings of `s`, then a second line with the running count after each prefix `s[0..i]`.

```cpp
// Reads one string s from stdin; prints the number of distinct non-empty
// palindromic substrings of s, then a second line with the running count after
// each prefix s[0..i]. Online via a palindromic tree (eertree).
#include <bits/stdc++.h>
using namespace std;

struct Eertree {
    enum { ODD = 0,   // imaginary root, len -1; seeds odd palindromes
           EVEN = 1 };// root, len 0; seeds even palindromes

    vector<int> s;              // processed characters; s[0] is a guard
    vector<int> len;            // node -> palindrome length
    vector<int> fail;           // node -> longest proper palindromic suffix
    vector<map<int,int>> go;    // node -> {char: child}
    int last;                   // node of the longest palindromic suffix so far
    long long num_distinct;     // distinct non-empty palindromes seen

    Eertree() {
        s.push_back(-1);        // guard
        len = {-1, 0};
        fail = {ODD, ODD};
        go.resize(2);
        last = EVEN;
        num_distinct = 0;
    }

    // climb suffix links until s can be wrapped: s[i-1-len] == s[i]
    int walk(int x) const {
        int i = (int)s.size() - 1;
        while (s[i - 1 - len[x]] != s[i]) x = fail[x];
        return x;
    }

    // append character c; returns true iff a new distinct palindrome was created
    bool add(int c) {
        s.push_back(c);
        int cur = walk(last);                // longest pal. suffix we can extend
        bool created = false;
        auto it = go[cur].find(c);
        if (it == go[cur].end()) {
            int now = (int)len.size();        // new node for c + (cur's pal.) + c
            len.push_back(len[cur] + 2);
            go.push_back(map<int,int>());
            if (len[now] == 1) {
                fail.push_back(EVEN);         // single char -> empty palindrome
            } else {
                int f = walk(fail[cur]);      // continue up fail[cur]
                fail.push_back(go[f][c]);
            }
            go[cur][c] = now;
            ++num_distinct;
            created = true;
        }
        last = go[cur][c];
        return created;
    }
};

int main() {
    string text;
    {
        // read the whole input as one token (first whitespace-delimited word)
        if (!(cin >> text)) text = "";
    }

    // map distinct characters to small integer codes (compact alphabet)
    map<char,int> code;
    for (char ch : text)
        if (!code.count(ch)) {
            int id = (int)code.size();
            code[ch] = id;
        }

    Eertree et;
    vector<long long> per_prefix;
    per_prefix.reserve(text.size());
    for (char ch : text) {
        et.add(code[ch]);
        per_prefix.push_back(et.num_distinct);
    }

    cout << et.num_distinct << "\n";
    for (size_t i = 0; i < per_prefix.size(); ++i)
        cout << per_prefix[i] << (i + 1 < per_prefix.size() ? ' ' : '\n');
    if (per_prefix.empty()) cout << "\n";
    return 0;
}
```
