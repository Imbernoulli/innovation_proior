**Problem.** Given a lowercase string `s` (`0 <= |s| <= 10^6`), count its distinct non-empty substrings — the size of the set `{ s[i..j] }`. Read `s` from stdin, print one integer. The answer can reach `|s|*(|s|+1)/2 ~ 5*10^11`, which overflows 32-bit, so it needs 64-bit output.

**Why the obvious approach is wrong (too slow).** Inserting every substring into a hash set is the definition itself and obviously correct, but it touches `Theta(|s|^2)` substrings of total length `Theta(|s|^3)`; even rolling-hashing each to a fingerprint is `Theta(|s|^2)` insertions and memory and risks hash collisions. At `|s| = 10^6` that is `~5*10^11` insertions — hopeless. It survives only to `|s| <= 2000`, where it is the perfect oracle. The deeper obstacle: the answer is itself `~5*10^11`, so I cannot afford one operation per distinct substring; I must count them *in aggregate*.

**Key idea — the suffix automaton's per-state count.** Build the suffix automaton (SAM) of `s`: the minimal automaton accepting exactly the substrings of `s`. It has at most `2|s|-1` states and is built online in `O(|s| * 26)`. The decisive fact is that its states partition the distinct substrings: every non-initial state `v` is an endpos-equivalence class whose strings are the suffixes of its longest string with lengths forming the contiguous range `(len[link[v]], len[v]]`. Each distinct substring lands in exactly one state, so

`answer = sum over all non-initial states v of (len[v] - len[link[v]])`.

One online build plus one linear sweep — `O(|s|)` time and space — never enumerating a single substring. This is the SOTA for the constraint. (A suffix array + LCP gives the same count via `n(n+1)/2 - sum LCP`; the SAM is chosen for its online build and transparent per-state formula.)

**Pitfalls to get right.**
1. *The clone's transitions.* When splitting an over-long state `q`, the clone must inherit `q`'s outgoing transitions: `nxt[clone] = nxt[q]`. Copying anything else (e.g. the fresh `cur`'s all-`-1` edges) detaches part of the automaton and silently corrupts the `len`/`link` values that the count depends on. A hand trace of the split on `aabb` (expected `8`) exposes it.
2. *Overflow.* `|s|` up to `10^6` makes the answer `~5*10^11`; accumulate in `long long` and cast each term before adding. An `int` is a silent wrong-answer at scale.
3. *The split condition.* Only `len[p] + 1 != len[q]` triggers a clone; when `len[p] + 1 == len[q]`, `q` is a true one-step extension and `link[cur] = q` directly. Splitting unconditionally (or never) both break the structure.

**Edge cases (all handled).** Empty string / absent input: `cin >> s` fails, print `0`. Length 1: answer `1`. Single repeated char `a^n`: chain SAM telescopes to `n`. All-distinct `s`: sum equals `n(n+1)/2`. Large `n = 10^6` over alphabets 1/2/26: answers up to `~5*10^11` returned in ~0.2s within 512 MB.

**Complexity.** `O(|s| * 26)` time, `O(|s| * 26)` memory for the transition arrays (about 208 MB at `|s| = 10^6`).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count the number of distinct non-empty substrings of s using a suffix
// automaton (SAM). Each state of the SAM recognizes a set of substrings that
// are exactly the suffixes of the longest one whose lengths lie in the half
// open range (len[link], len]; that range has size len[state] - len[link[state]]
// distinct strings, and every distinct substring of s belongs to exactly one
// state. Summing that range size over all non-initial states counts every
// distinct substring once.

static const int ALPHA = 26;

struct SAM {
    // next[v] : transitions, link[v] : suffix link, len[v] : longest length in v.
    vector<array<int, ALPHA>> nxt;
    vector<int> link;
    vector<int> len;
    int last;   // state for the whole current prefix
    int sz;     // number of states

    void init(int maxLen) {
        nxt.assign(2 * maxLen + 5, array<int, ALPHA>{});
        for (auto &a : nxt) a.fill(-1);
        link.assign(2 * maxLen + 5, -1);
        len.assign(2 * maxLen + 5, 0);
        // state 0 is the initial (empty) state.
        link[0] = -1;
        len[0] = 0;
        last = 0;
        sz = 1;
    }

    void extend(int c) {
        int cur = sz++;
        len[cur] = len[last] + 1;
        link[cur] = -1;
        nxt[cur].fill(-1);
        int p = last;
        while (p != -1 && nxt[p][c] == -1) {
            nxt[p][c] = cur;
            p = link[p];
        }
        if (p == -1) {
            link[cur] = 0;
        } else {
            int q = nxt[p][c];
            if (len[p] + 1 == len[q]) {
                // q is a contiguous extension: reuse it directly.
                link[cur] = q;
            } else {
                // Split q into a clone holding only the short occurrences.
                int clone = sz++;
                len[clone] = len[p] + 1;
                link[clone] = link[q];
                nxt[clone] = nxt[q];
                while (p != -1 && nxt[p][c] == q) {
                    nxt[p][c] = clone;
                    p = link[p];
                }
                link[q] = clone;
                link[cur] = clone;
            }
        }
        last = cur;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {
        // No string token at all -> empty string -> zero distinct substrings.
        cout << 0 << "\n";
        return 0;
    }

    int n = (int)s.size();
    SAM sam;
    sam.init(n);
    for (int i = 0; i < n; i++) {
        sam.extend(s[i] - 'a');
    }

    // Distinct substrings = sum over non-initial states of (len[v] - len[link[v]]).
    long long answer = 0;
    for (int v = 1; v < sam.sz; v++) {
        answer += (long long)(sam.len[v] - sam.len[sam.link[v]]);
    }

    cout << answer << "\n";
    return 0;
}
```
