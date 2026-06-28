# Suffix Automaton (SAM) — online distinct-substring count

## Problem

Given a string $s$ of length $n$, count the number of distinct non-empty substrings of $s$, with $s$ built one character at a time (online), in $O(n)$ over a fixed alphabet.

## Key idea

Build the **suffix automaton**: the minimal deterministic automaton for the suffixes of $s$. Every substring of $s$ is a prefix of some suffix, so it is exactly a path label starting from the initial state $t_0$. Distinct substrings then equal distinct non-empty paths from $t_0$.

**Endpos equivalence (the merge rule).** For a substring $t$, let $\mathrm{endpos}(t)$ be its set of ending positions in $s$. Two substrings share one state iff they have the same $\mathrm{endpos}$ — they are then followed by the same continuations everywhere, hence interchangeable. The automaton has one state per $\mathrm{endpos}$ class plus $t_0$.

**Structure of a class.** Endpos sets are *laminar*: for $|u|\le|w|$, either $u$ is a suffix of $w$ and $\mathrm{endpos}(w)\subseteq\mathrm{endpos}(u)$, or the sets are disjoint. Within one class the strings are exactly the suffixes of the longest string $\mathrm{longest}(v)$ over a contiguous length interval $[\mathrm{minlen}(v),\mathrm{len}(v)]$. So state $v$ owns $\mathrm{len}(v)-\mathrm{minlen}(v)+1$ distinct substrings.

**Suffix link.** $\mathrm{link}(v)$ is the state of the longest proper suffix of $\mathrm{longest}(v)$ lying in a *different* endpos class. Then $\mathrm{minlen}(v)=\mathrm{len}(\mathrm{link}(v))+1$, and the links form a tree rooted at $t_0$ (the inclusion tree of endpos sets).

**Distinct-substring count.** Summing the per-state counts and substituting $\mathrm{minlen}(v)=\mathrm{len}(\mathrm{link}(v))+1$:

$$\#\{\text{distinct non-empty substrings}\} \;=\; \sum_{v \ne t_0}\big(\mathrm{len}(v)-\mathrm{len}(\mathrm{link}(v))\big).$$

One sweep over the states gives the final count. Online, maintain the same sum incrementally: after `extend(c)`, the net new contribution is $\mathrm{len}(cur)-\mathrm{len}(\mathrm{link}(cur))$. If a clone is created, it only partitions $q$'s old length interval, so $q$ plus `clone` contributes exactly what $q$ contributed before.

## Online construction — `extend(c)`

Appending $c$ to $s$ injects exactly the strings $x+c$ for every suffix $x$ of $s$ (including $c$ itself). Let $last$ be the state of the whole string $s$.

1. Create $cur$ with $\mathrm{len}(cur)=\mathrm{len}(last)+1$ (the new whole string $s+c$).
2. Walk suffix links from $last$. While the current state $p$ has no $c$-transition, add $p\xrightarrow{c}cur$ and follow $\mathrm{link}(p)$. Stop at the first $p$ that already has a $c$-edge, or when falling off the top ($p=-1$).
3. If $p=-1$: $c$ is new, set $\mathrm{link}(cur)=t_0$.
4. Else let $q$ be the target of $p\xrightarrow{c}q$.
   - **Tight edge** $\mathrm{len}(p)+1=\mathrm{len}(q)$: $q$'s longest string is exactly $x+c$. Set $\mathrm{link}(cur)=q$.
   - **Loose edge** $\mathrm{len}(p)+1<\mathrm{len}(q)$: $x+c$ gains the new ending position but $q$'s longer strings do not, so $q$ splits. **Clone** $q$ into $clone$ — copy its transitions and its link, set $\mathrm{len}(clone)=\mathrm{len}(p)+1$. Walking up from $p$, redirect every $c$-edge that pointed to $q$ to $clone$. Set $\mathrm{link}(q)=\mathrm{link}(cur)=clone$.
5. Add $\mathrm{len}(cur)-\mathrm{len}(\mathrm{link}(cur))$ to the running answer, then set $last=cur$.

## $O(n)$ size and time

- **States** $\le 2n-1$ ($n\ge2$): each character adds one $cur$ and at most one $clone$. Independently, the endpos inclusion tree has at most $n$ leaves; after suppressing single-child internal nodes, every internal node has degree at least two, so the tree has at most $2n-1$ nodes.
- **Transitions** $O(n)$: tight edges form a spanning tree of longest paths, and loose edges can be charged injectively to proper suffixes.
- **Time** $O(n)$ over a fixed alphabet. The edge-adding walk creates one transition per non-stopping step; cloning copies $O(1)$ outgoing entries per clone; the redirect walk is amortized by the standard monotone suffix-position argument. With fixed transition arrays this is worst-case $O(n)$ time and $O(nk)$ memory for alphabet size $k$; the dictionary version below is expected $O(n)$ for a fixed alphabet.

## Code

Single-file C++17. It reads one string `s` from stdin and prints, on the first line, the number of distinct non-empty substrings of `s`; on the second line, the running count after each prefix `s[0..i]` (space-separated). The total is accumulated in `long long` (it grows like $\Theta(n^2)$, so it overflows 32 bits well before $n=10^6$).

```cpp
// Online suffix automaton (SAM): counts distinct non-empty substrings.
// Reads one string s from stdin; prints the number of distinct non-empty
// substrings of s on the first line, then the running count after each prefix
// s[0..i] (space-separated) on the second line.
#include <bits/stdc++.h>
using namespace std;

// One state per ending-position equivalence class (plus the root t0 = state 0).
// State v stores len[v] (length of its longest string), link[v] (the suffix
// link: the state of the longest proper suffix of longest(v) lying in another
// class), and trans[v] (character code -> state). The distinct non-empty
// substrings owned by v are exactly the suffixes of longest(v) with lengths in
// [len[link[v]] + 1, len[v]], i.e. len[v] - len[link[v]] of them.
struct SubstringCounter {
    vector<long long> length;       // len[v]: length of v's longest string
    vector<int> link;               // link[v]: suffix link (-1 for the root)
    vector<map<int, int>> trans;    // trans[v]: character code -> state
    int last;                       // state whose longest string is the whole prefix
    long long total;                // current number of distinct non-empty substrings

    SubstringCounter() {
        length.push_back(0);        // len of the root's longest string (empty) is 0
        link.push_back(-1);         // root has no suffix link
        trans.emplace_back();       // root's outgoing transitions
        last = 0;
        total = 0;
    }

    void extend(int c) {
        // New state for the whole prefix s + c; its longest string grew by one.
        int cur = (int)length.size();
        length.push_back(length[last] + 1);
        link.push_back(-1);
        trans.emplace_back();

        // Walk suffix links from last, attaching a c-edge to cur for every
        // suffix of s that was not yet followed by c (those x + c are new).
        int p = last;
        while (p != -1 && trans[p].find(c) == trans[p].end()) {
            trans[p][c] = cur;
            p = link[p];
        }

        if (p == -1) {
            // c never followed any suffix of s: c is new, link cur to the root.
            link[cur] = 0;
        } else {
            int q = trans[p][c];
            if (length[p] + 1 == length[q]) {
                // Tight edge: q's longest string is exactly x + c. Link straight.
                link[cur] = q;
            } else {
                // Loose edge: x + c gained the new ending position but q's longer
                // strings did not, so q's class splits. Clone q at length len(p)+1.
                int clone = (int)length.size();
                length.push_back(length[p] + 1);
                link.push_back(link[q]);     // clone copies q's link
                trans.push_back(trans[q]);   // and q's transitions
                // Redirect the suffix chain of c-edges from q over to the clone.
                while (p != -1) {
                    auto it = trans[p].find(c);
                    if (it == trans[p].end() || it->second != q) break;
                    it->second = clone;
                    p = link[p];
                }
                link[q] = clone;
                link[cur] = clone;
            }
        }

        last = cur;
        total += length[cur] - length[link[cur]];
    }

    long long count_distinct_substrings() const { return total; }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    // Read the string as a single whitespace-delimited token.
    if (!(cin >> s)) { cout << 0 << "\n"; return 0; }

    // Assign stable dense integer codes to characters as they first appear,
    // then feed one code at a time (the alphabet is fixed for a given input).
    SubstringCounter sc;
    vector<long long> perPrefix;
    perPrefix.reserve(s.size());
    int code[256];
    fill(begin(code), end(code), -1);
    int nextCode = 0;

    for (unsigned char ch : s) {
        if (code[ch] == -1) code[ch] = nextCode++;
        sc.extend(code[ch]);
        perPrefix.push_back(sc.count_distinct_substrings());
    }

    cout << sc.count_distinct_substrings() << "\n";
    for (size_t i = 0; i < perPrefix.size(); i++) {
        cout << perPrefix[i] << (i + 1 < perPrefix.size() ? ' ' : '\n');
    }
    return 0;
}
```
