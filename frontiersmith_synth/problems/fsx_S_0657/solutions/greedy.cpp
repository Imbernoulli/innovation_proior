// TIER: greedy
// "greedy-repair-search": scan every cell once, heaviest weight first; accept a candidate
// into the running mark set iff it keeps the set Sidon-feasible on the torus, otherwise
// discard it for good. This is the obvious first approach an average strong coder writes --
// a single myopic pass with no notion of the underlying algebraic (difference-set) structure
// -- and it plateaus well short of the algebraic ceiling on the planted tests, where it burns
// its early picks on a cluster of mutually-conflicting heavy cells.
#include <bits/stdc++.h>
using namespace std;

static int H, W;

bool isFeasible(const vector<pair<int, int>> &marks) {
    int m = marks.size();
    set<long long> diffs;
    for (int a = 0; a < m; a++)
        for (int b = 0; b < m; b++) {
            if (a == b) continue;
            int dr = ((marks[a].first - marks[b].first) % H + H) % H;
            int dc = ((marks[a].second - marks[b].second) % W + W) % W;
            long long id = (long long)dr * W + dc;
            if (!diffs.insert(id).second) return false;
        }
    return true;
}

int main() {
    long long lnum, lden;
    cin >> H >> W >> lnum >> lden;
    vector<vector<long long>> w(H, vector<long long>(W));
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) cin >> w[i][j];

    vector<tuple<long long, int, int>> cells;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) cells.push_back({w[i][j], i, j});
    sort(cells.begin(), cells.end(), [](auto &a, auto &b) {
        if (get<0>(a) != get<0>(b)) return get<0>(a) > get<0>(b);
        if (get<1>(a) != get<1>(b)) return get<1>(a) < get<1>(b);
        return get<2>(a) < get<2>(b);
    });

    vector<pair<int, int>> marks;
    for (auto &[wt, r, c] : cells) {
        vector<pair<int, int>> trial = marks;
        trial.push_back({r, c});
        if (isFeasible(trial)) marks = trial; // accept; else discard for good, no repair
    }

    cout << marks.size() << "\n";
    for (auto &mk : marks) cout << mk.first << " " << mk.second << "\n";
    return 0;
}
