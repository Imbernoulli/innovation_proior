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
