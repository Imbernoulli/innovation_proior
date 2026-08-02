// TIER: greedy
// The "obvious" approach: this LOOKS like a single combinatorial game (one graph, several
// tokens), so just minimax it directly. Search the JOINT state space (a tuple of all c
// token positions) with memoization, capped by an expansion budget so it terminates. Two
// naive shortcuts, typical of a first attempt that has not seen Sprague-Grundy theory:
//   (1) a joint state revisited while still "in progress" on the recursion stack (a genuine
//       cycle in the joint-move-graph, which the CYCLIC components are built to create) is
//       just assumed to be a LOSS for whoever is on move there, instead of being handled
//       correctly;
//   (2) if the expansion budget runs out, give up entirely and PASS.
// This never realizes the position decomposes into independent components, so the joint
// state space it searches is the PRODUCT of every component's size -- exponential in the
// number of components c. It matches the exact answer whenever that product is small, and
// increasingly misses winning moves (falls back to PASS, or loses track via shortcut (1))
// as c and component sizes grow.
#include <bits/stdc++.h>
using namespace std;

static const long long CAP = 4500;

struct VectorHash {
    size_t operator()(const vector<int>& v) const {
        size_t h = v.size();
        for (int x : v) h ^= (std::hash<int>()(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        return h;
    }
};

int c_g;
vector<vector<int>> adj_g;
unordered_map<vector<int>, int, VectorHash> status_g; // 1=in-progress, 2=WIN, 3=LOSE
long long expandCount_g;
bool aborted_g;

int solve(const vector<int>& state) {
    if (aborted_g) return -1;
    auto it = status_g.find(state);
    if (it != status_g.end()) {
        if (it->second == 1) return 3; // naive shortcut: on-path repeat treated as LOSE
        return it->second;
    }
    expandCount_g++;
    if (expandCount_g > CAP) { aborted_g = true; return -1; }
    status_g[state] = 1;
    bool anyMove = false;
    int outcome = 3;
    for (int i = 0; i < c_g && !aborted_g; i++) {
        for (int nb : adj_g[state[i]]) {
            anyMove = true;
            vector<int> ns = state; ns[i] = nb;
            int r = solve(ns);
            if (aborted_g) break;
            if (r == 3) { outcome = 2; goto done; }
        }
    }
done:
    if (aborted_g) return -1;
    if (!anyMove) outcome = 3;
    status_g[state] = outcome;
    return outcome;
}

int main() {
    long long M;
    if (!(cin >> M)) return 0;
    for (long long m = 0; m < M; m++) {
        int n, c, e;
        cin >> n >> c >> e;
        vector<int> tokens(c);
        for (int i = 0; i < c; i++) cin >> tokens[i];
        adj_g.assign(n, {});
        for (int i = 0; i < e; i++) {
            int u, v; cin >> u >> v;
            adj_g[u].push_back(v);
        }
        c_g = c;
        status_g.clear();
        expandCount_g = 0;
        aborted_g = false;

        int rootOutcome = solve(tokens);
        pair<int,int> answer = {-1, -1};
        if (!aborted_g && rootOutcome == 2) {
            for (int i = 0; i < c && answer.first == -1; i++) {
                for (int nb : adj_g[tokens[i]]) {
                    vector<int> ns = tokens; ns[i] = nb;
                    auto it = status_g.find(ns);
                    if (it != status_g.end() && it->second == 3) { answer = {i, nb}; break; }
                }
            }
        }
        if (aborted_g || rootOutcome != 2 || answer.first == -1) cout << "PASS\n";
        else cout << "MOVE " << tokens[answer.first] << " " << answer.second << "\n";
    }
    return 0;
}
