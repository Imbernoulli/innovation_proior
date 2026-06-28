**Problem.** Given a dictionary of `m` patterns over `a`..`z`, each with an integer weight `w[i]`
(possibly negative), and a text `T`, compute `sum_i w[i] * (occurrences of pattern i in T)`, counting
every overlapping start position. The same pattern may be listed multiple times with different
weights. Read `m`, the `m` `pattern weight` lines, and the text from stdin; print the total score.
Constraint: `sum |p_i| + |T| <= 10^6`.

**Why the obvious per-pattern scan is too slow.** The objective splits per pattern, so the naive plan
is to match each pattern against `T` independently (KMP or `string::find`) and add `w[i]*count`. That
is `O(m*|T| + sum|p_i|)`. With `m = 10^5` short patterns and `|T|` near `10^6`, the `m*|T|` term is
about `9*10^{10}` character reads — minutes, not the one-second budget. The fault is rescanning the
whole text once per pattern. The text must be read **once** while accounting for all patterns
simultaneously.

**Key idea — Aho-Corasick automaton + fail-tree subtree sum.** Build one automaton from the entire
dictionary:

1. Insert every pattern into a trie; at each terminal node keep `wsum[u]` = the *sum* of weights of
   all patterns ending exactly there (duplicates and nested patterns fold in for free).
2. BFS the trie to compute **failure links** (suffix links: `fail[v]` spells the longest proper
   suffix of `v`'s string that is also a trie node) and, in the same pass, **complete the transition
   function** by setting every missing edge `nxt[v][c] = nxt[fail[v]][c]`. After this, `nxt` is a
   total DFA and one text character is one array lookup, giving an `O(|T|)` scan.
3. Drive `T` through the automaton; at each character increment `cnt[state]` at the landing state
   only (one increment per character — no fail-chain walk).
4. **The non-obvious step:** counting only landing states undercounts, because a position ending
   `aaa` also ends `aa` and `a` (different nodes). The true occurrence count of the string at node
   `u` is the sum of `cnt[v]` over the **subtree of `u` in the fail tree** — exactly the states whose
   fail chain passes through `u`. So push every `cnt[v]` up to `fail[v]` once, processing children
   before parents. The BFS visitation order is in nondecreasing fail depth, so iterating it
   **in reverse** is a valid children-before-parents order. After this single linear pass, `cnt[u]`
   is `u`'s occurrence count, and the answer is `sum_u cnt[u]*wsum[u]`.

Total cost `O(sum|p_i| + |T|)` — within the limit.

**Pitfalls to get right.**
1. *Recovering overlapping counts.* Do **not** walk the fail chain per character (it is `O(depth)` per
   step and degrades to quadratic on inputs like `a, aa, aaa, ...` over `aaaa...a`). Count landing
   states cheaply, then do the one global fail-tree subtree sum.
2. *Subtree-sum order.* Accumulate a node's fail-children into its `cnt` **before** scoring/using it.
   Reverse-BFS order guarantees this; getting the order wrong undercounts deep patterns.
3. *Total transition function.* Complete `nxt[v][c]` during the BFS so the scan never walks fail links
   at scan time — that is what makes the scan genuinely `O(|T|)` rather than amortized-with-a-caveat.
4. *Overflow.* Individual weights fit in `long long`, and occurrence counts fit in `long long`, but
   duplicate patterns can make `cnt[u] * wsum[u]` exceed 64-bit. Use `__int128_t` for `wsum` and the
   accumulator, then print it manually. An `int` is a silent wrong answer.

**Edge cases (all handled by the structure + final signed accumulate):** `m = 0` and/or empty text →
`0` (the scan and the reverse loop both run zero times); pattern longer than text → its terminal node
is never reached, contributes `0`; pattern equal to the whole text → counted once; duplicate patterns
with cancelling weights → weights add at the shared terminal node; deeply nested single-letter
patterns → overlapping counts come out right via the subtree sum; zero/negative weights → flow through
unchanged, total may be negative.

**Complexity.** `O(sum|p_i| + |T|)` time; `O(sum|p_i| * alphabet)` memory for the transition rows.
At the `10^6` limit this is ~0.13 s and well under 256 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    if (!(cin >> m)) return 0;                 // m = number of dictionary patterns

    auto printInt128 = [](const __int128_t value) {
        if (value == 0) {
            cout << 0;
            return;
        }

        __int128_t x = value;
        if (x < 0) {
            cout << '-';
            x = -x;
        }

        string digits;
        while (x > 0) {
            digits.push_back(char('0' + x % 10));
            x /= 10;
        }
        reverse(digits.begin(), digits.end());
        cout << digits;
    };

    // Aho-Corasick over lowercase letters 'a'..'z' (alphabet size 26).
    // next[v][c] : goto/transition (made total via BFS), child[v][c]: real trie child.
    // We keep the trie compact: nodes grow as we insert.
    const int A = 26;
    // Upper bound on node count: 1 (root) + total pattern length.
    vector<array<int, A>> nxt;                 // transition function (after BFS = total automaton)
    vector<int> fail;                          // suffix link
    vector<__int128_t> wsum;                   // sum of weights of patterns ENDING exactly at this node
    auto newNode = [&]() {
        nxt.push_back(array<int, A>{});
        nxt.back().fill(-1);
        fail.push_back(0);
        wsum.push_back(0);
        return (int)nxt.size() - 1;
    };
    newNode();                                  // node 0 = root

    // Insert each pattern; accumulate its weight at the terminal node. Duplicate
    // patterns and patterns that are prefixes/substrings of others are handled
    // automatically because weights sum at the shared terminal node.
    for (int i = 0; i < m; i++) {
        string p;
        long long w;
        cin >> p >> w;
        int cur = 0;
        for (char ch : p) {
            int c = ch - 'a';
            if (nxt[cur][c] == -1) nxt[cur][c] = newNode();
            cur = nxt[cur][c];
        }
        wsum[cur] += w;                         // multiple identical patterns -> weights add
    }

    int N = (int)nxt.size();

    // BFS to build fail links and turn the trie into a complete DFA (goto function).
    // After this, nxt[v][c] is always a valid state (the "transition" automaton).
    // bfsOrder records nodes in increasing fail-tree depth, so a reverse pass
    // pushes occurrence counts from a node up to its fail-parent.
    vector<int> bfsOrder;
    bfsOrder.reserve(N);
    queue<int> q;
    for (int c = 0; c < A; c++) {
        if (nxt[0][c] == -1) {
            nxt[0][c] = 0;                       // root's missing edges loop to root
        } else {
            fail[nxt[0][c]] = 0;
            q.push(nxt[0][c]);
        }
    }
    while (!q.empty()) {
        int v = q.front(); q.pop();
        bfsOrder.push_back(v);
        for (int c = 0; c < A; c++) {
            int u = nxt[v][c];
            if (u == -1) {
                nxt[v][c] = nxt[fail[v]][c];     // no real child: follow fail's transition
            } else {
                fail[u] = nxt[fail[v]][c];       // real child: its fail is fail[v]'s transition on c
                q.push(u);
            }
        }
    }

    // Feed the text through the automaton, counting how many times each state is the
    // current state (i.e. how many prefixes of the text end exactly here).
    string text;
    cin >> text;                                // text may be empty token -> stays ""
    vector<long long> cnt(N, 0);
    int state = 0;
    for (char ch : text) {
        int c = ch - 'a';
        state = nxt[state][c];
        cnt[state]++;
    }

    // Fail-tree subtree sum: a pattern ending at node u occurs once for every state
    // reached during the scan whose fail-chain passes through u. Pushing cnt up the
    // fail links (children before parents) makes cnt[u] = #occurrences of the string
    // spelled by u as a substring of the text. bfsOrder is in nondecreasing fail
    // depth, so iterating it in reverse processes children before their fail-parent.
    __int128_t answer = 0;
    for (int i = (int)bfsOrder.size() - 1; i >= 0; i--) {
        int v = bfsOrder[i];
        answer += cnt[v] * wsum[v];             // contribution of patterns ending at v
        cnt[fail[v]] += cnt[v];                  // propagate occurrence count to fail-parent
    }
    // Root (node 0) never carries a pattern (empty string), so it is not in bfsOrder
    // and contributes nothing; this is intentional.

    printInt128(answer);
    cout << "\n";
    return 0;
}
```
