// TIER: greedy
// The "standard recipe": greedy nearest-neighbour chaining. Start at guest 1
// and repeatedly walk to whichever UNVISITED guest you have exchanged the
// most words with (aggregated exchange weight from the current seat); when no
// exchange partner remains unvisited, seat the next unseated guest in id
// order and continue from there. This is the natural first-attempt heuristic
// for "seat frequent talkers together" -- it never looks at the clan label,
// it only ever reacts to the local, already-visited-filtered edge weights.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, K, M;
    if (!(cin >> N >> K >> M)) return 0;
    vector<int> phase(N + 1);
    for (int i = 1; i <= N; i++) cin >> phase[i];

    vector<unordered_map<int, ll>> w(N + 1);
    for (int i = 0; i < M; i++) {
        int u, v, c; cin >> u >> v >> c;
        w[u][v] += c;
        w[v][u] += c;
    }

    vector<char> visited(N + 1, 0);
    vector<int> result;
    result.reserve(N);

    int tail = 1;
    visited[tail] = 1;
    result.push_back(tail);
    int idCursor = 1;

    while ((int)result.size() < N) {
        int best = -1; ll bestW = -1;
        for (auto& pr : w[tail]) {
            int v = pr.first; ll ww = pr.second;
            if (visited[v]) continue;
            if (ww > bestW || (ww == bestW && v < best)) { bestW = ww; best = v; }
        }
        if (best == -1) {
            while (idCursor <= N && visited[idCursor]) idCursor++;
            if (idCursor > N) break;
            best = idCursor;
        }
        visited[best] = 1;
        result.push_back(best);
        tail = best;
    }
    for (int g = 1; g <= N; g++) if (!visited[g]) result.push_back(g);

    for (size_t i = 0; i < result.size(); i++) cout << result[i] << (i + 1 < result.size() ? ' ' : '\n');
    return 0;
}
