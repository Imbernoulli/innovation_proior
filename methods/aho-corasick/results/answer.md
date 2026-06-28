# Aho-Corasick multi-keyword matching

## Problem

Given a finite set of nonempty keywords `K = {y1, ..., yk}` and a text `x` of
length `n`, report every occurrence of every keyword in `x`, overlaps included.
The straightforward method scans the text once per keyword and costs
`Theta(k*n)`; the goal is to read the text once, with non-output work independent
of `k`.

## Key idea

Build one finite-state pattern matching machine for the whole keyword set, then
run the text through it once. The machine has three pieces:

- **goto** `g(s, a)` is the trie of all keywords. The path from the root to a
  state spells a prefix of some keyword. Missing edges mean failure, while a
  missing edge at the root acts like a self-loop to the root.
- **failure** `f(s)` generalizes the Knuth-Morris-Pratt failure function from one
  pattern to a trie. If state `s` spells `u`, then `f(s)` is the state spelling
  the longest proper suffix of `u` that is also a prefix of some keyword.
- **output links** keep reporting separate from failure recovery. `output(s)`
  stores only keywords ending exactly at state `s`; `out_link(s)` points to the
  nearest failure ancestor with a nonempty local output.

Search keeps a current state. For each text symbol, follow failure links until a
goto edge is available, take that edge, emit local outputs of the new state, then
walk output links to emit terminal suffix states.

The optional deterministic next-move table over a chosen alphabet is filled in
breadth-first order: `delta(0, a) = g(0, a)`, and for a non-root state `r`,
`delta(r, a) = g(r, a)` when that goto exists, otherwise
`delta(r, a) = delta(f(r), a)`.

## Correctness

- The failure construction is correct by induction on trie depth: for a child
  `s = g(r, a)`, the candidates for a proper suffix of `s` are obtained by
  walking `f(r), f(f(r)), ...` and extending the first state that has a goto on
  `a`.
- The output-link chain from `s` visits exactly the terminal states on the
  failure chain of `s`, so local outputs plus output-link outputs are precisely
  the keywords whose strings are suffixes of the string for `s`.
- After processing text prefix `x[:j]`, the current state spells the longest
  suffix of that prefix that is also a keyword prefix. Therefore the emitted
  outputs at position `j` are exactly the keywords ending there.

## Complexity

Construction is linear in the total keyword length: trie insertion visits each
keyword character once, the failure-link construction has an amortized linear
number of failure steps charged over keyword paths, and assigning `out_link` is
constant time per trie edge because the failure ancestor has already been
processed in breadth-first order.

During search, each input symbol causes exactly one goto transition and the total
number of failure transitions over the whole scan is less than `n`, because every
failure lowers trie depth and gotos can raise it only by one. Non-output search
therefore takes fewer than `2n` state transitions, plus the unavoidable cost of
emitting the matches.

Using the deterministic next-move table replaces the goto/failure loop with one
transition per input symbol, at the cost of storing more transitions.

## Algorithm

Single-file C++17. It reads `k`, then `k` keywords, then the text, and prints
every `start_index keyword` match (0-indexed start, overlaps included), sorted by
`(start, keyword)`. The build folds failure recovery into `go`, so each text
symbol is one deterministic transition.

```cpp
#include <bits/stdc++.h>
using namespace std;

struct AhoCorasick {
    vector<array<int, 256>> go;     // goto: go[s][c] = next state, -1 = fail
    vector<vector<int>> output;     // indices of keywords ending exactly at s
    vector<int> fail;               // failure link
    vector<int> out_link;           // nearest terminal state on the failure chain
    vector<string> keywords;        // stored keywords, by index

    AhoCorasick() { new_state(); }  // state 0 is the root

    int new_state() {
        go.push_back({});
        go.back().fill(-1);
        output.emplace_back();
        fail.push_back(0);
        out_link.push_back(0);
        return (int)go.size() - 1;
    }

    void add_keyword(const string& word, int id) {
        int state = 0;
        for (unsigned char ch : word) {
            if (go[state][ch] == -1) go[state][ch] = new_state();
            state = go[state][ch];
        }
        output[state].push_back(id);
    }

    void build() {
        queue<int> q;
        for (int c = 0; c < 256; ++c) {
            int s = go[0][c];
            if (s == -1) {
                go[0][c] = 0;          // root self-loops on unmatched symbols
            } else {
                fail[s] = 0;
                q.push(s);
            }
        }
        while (!q.empty()) {
            int r = q.front(); q.pop();
            for (int c = 0; c < 256; ++c) {
                int s = go[r][c];
                if (s == -1) {
                    go[r][c] = go[fail[r]][c];   // build the deterministic move
                    continue;
                }
                q.push(s);
                fail[s] = go[fail[r]][c];        // fail[r] already deterministic
                out_link[s] = output[fail[s]].empty() ? out_link[fail[s]]
                                                       : fail[s];
            }
        }
    }

    // Emit every (start, keyword-id) match, overlaps included, in one pass.
    void search(const string& text, vector<pair<int, int>>& matches) const {
        int state = 0;
        for (int i = 0; i < (int)text.size(); ++i) {
            state = go[state][(unsigned char)text[i]];
            for (int id : output[state])
                matches.push_back({i - (int)keywords[id].size() + 1, id});
            for (int out = out_link[state]; out; out = out_link[out])
                for (int id : output[out])
                    matches.push_back({i - (int)keywords[id].size() + 1, id});
        }
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int k;
    if (!(cin >> k)) return 0;

    AhoCorasick ac;
    ac.keywords.resize(k);
    for (int i = 0; i < k; ++i) {
        cin >> ac.keywords[i];
        ac.add_keyword(ac.keywords[i], i);
    }
    ac.build();

    string text;
    cin >> text;

    vector<pair<int, int>> matches;          // (start, keyword-id)
    ac.search(text, matches);

    sort(matches.begin(), matches.end(),
         [&](const pair<int, int>& a, const pair<int, int>& b) {
             if (a.first != b.first) return a.first < b.first;
             return ac.keywords[a.second] < ac.keywords[b.second];
         });

    for (auto& m : matches)
        cout << m.first << ' ' << ac.keywords[m.second] << '\n';
    return 0;
}
```

On `{he, she, his, hers}` over `"ushers"` (input `4 / he / she / his / hers /
ushers`) the machine reports `1 she`, `2 he`, and `2 hers`: every keyword
substring, overlaps included, in one pass.
