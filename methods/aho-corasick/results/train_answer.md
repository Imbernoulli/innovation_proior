The problem is to find every occurrence of every keyword in a long text, including overlapping matches, without paying a cost proportional to the number of keywords. The obvious approach scans the text once per keyword, so a query with dozens of keywords reads the same corpus dozens of times. Knuth–Morris–Pratt solves the single-pattern case in linear time, but running it independently for each keyword simply rebuilds the same k-times-text cost. A trie matches all keywords that start at the same text position simultaneously, yet when the next character has no outgoing edge the trie offers no rule for recovering the partially matched prefix; one must restart at every position. Building a full deterministic automaton by determinizing a regular expression gives one transition per symbol, but the subset construction can blow up exponentially in the size of the pattern set and is complicated to implement. What is missing is a structure that has the trie's shared-prefix matching and KMP's failure recovery at the same time.

The method that supplies both is the Aho–Corasick algorithm. It builds a single finite-state machine for the entire keyword set and scans the text once. The first component is the goto function, which is just the trie of all keywords: shared prefixes share nodes, and each node reached by a complete keyword is tagged with that keyword. The second component is a failure link for every node. If a node represents the string u, its failure link points to the deepest node whose string is a proper suffix of u and is also a prefix of some keyword. This directly generalizes the KMP failure function from a single pattern to a set of patterns. The third component is an output link: from each node, an output link jumps to the nearest failure ancestor that actually ends a keyword, so suffix keywords can be reported without walking through empty failure states one by one.

Searching is then a single left-to-right pass. Maintain a current state. For each input character, follow failure links until the current state has a goto edge for that character, then take that edge. Because the root has an implicit self-loop on every missing character, this loop always terminates. After moving, emit every keyword stored at the new state, then follow output links and emit their keywords as well. Each text character causes exactly one goto transition, and the total number of failure transitions over the whole scan is bounded by the text length, so the non-output work is linear in the text length and does not depend on the number of keywords. Construction is also linear in the total keyword length: the trie is built by inserting each keyword once, and the failure links are computed in breadth-first order so that when a node's failure link is needed, its target has already been processed.

Concretely, as a single-file C++17 program that reads `k`, then `k` keywords, then the text from standard input and prints every `start_index keyword` match (0-indexed start, overlaps included), sorted by `(start, keyword)`. I fold the failure recovery directly into the `go` table during the breadth-first build, so the inner search loop is a single deterministic transition per symbol; the output links still let suffix keywords be reported without walking empty failure states.

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

On the keyword set `{he, she, his, hers}` over the text `ushers` (input `4 / he /
she / his / hers / ushers`), this prints `1 she`, `2 he`, and `2 hers`: every
keyword substring, overlaps included, found in a single pass whose cost does not
depend on the number of keywords.
