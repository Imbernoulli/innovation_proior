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
