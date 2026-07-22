// TIER: greedy
// The natural "try harder" recipe: a plain greedy WALK (append the smallest legal,
// not-yet-used symbol) from many candidate start vertices, keeping the longest walk
// found.  This is the obvious improvement over the fixed-start baseline -- but it is
// still a SINGLE trail: on a balanced component it returns to its start and stalls,
// leaving side-cycles uncovered.  Without the Eulerian cycle-splicing insight it
// cannot cover the whole component, so it falls well short of the strong solution.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int A, K, V;
static ll E;
static vector<char> forb;

int main() {
    int F;
    scanf("%d %d %d", &A, &K, &F);
    V = 1; for (int i = 0; i < K - 1; i++) V *= A;
    E = (ll)V * A;

    forb.assign(E, 0);
    static char buf[64];
    for (int t = 0; t < F; t++) {
        scanf("%s", buf);
        ll code = 0;
        for (int i = 0; i < K; i++) code = code * A + (buf[i] - '0');
        forb[code] = 1;
    }

    // candidate starts: every clean vertex if there are few, else a strided sample.
    vector<int> clean;
    for (int v = 0; v < V; v++)
        for (int s = 0; s < A; s++)
            if (!forb[(ll)v * A + s]) { clean.push_back(v); break; }
    if (clean.empty()) { printf("0\n"); return 0; }

    vector<int> starts;
    int want = 64;
    if ((int)clean.size() <= want) starts = clean;
    else {
        int stride = (int)clean.size() / want;
        for (size_t i = 0; i < clean.size() && (int)starts.size() < want; i += stride)
            starts.push_back(clean[i]);
    }

    string best;
    ll bestCov = -1;
    vector<char> used(E, 0);

    for (int st : starts) {
        fill(used.begin(), used.end(), 0);
        string S;
        {
            vector<int> digs(K - 1, 0);
            int x = st;
            for (int i = K - 2; i >= 0; i--) { digs[i] = x % A; x /= A; }
            for (int i = 0; i < K - 1; i++) S.push_back((char)('0' + digs[i]));
        }
        int cur = st;
        ll cov = 0;
        while (true) {
            // dead-end avoidance: among legal unused out-edges, take the one whose
            // target still has the most unused out-edges (ties -> smallest symbol).
            int bestS = -1, bestScore = -1;
            for (int s = 0; s < A; s++) {
                ll ec = (ll)cur * A + s;
                if (forb[ec] || used[ec]) continue;
                int nx = (int)(ec % V);
                int sc = 0;
                for (int s2 = 0; s2 < A; s2++) {
                    ll e2 = (ll)nx * A + s2;
                    if (!forb[e2] && !used[e2]) sc++;
                }
                if (sc > bestScore) { bestScore = sc; bestS = s; }
            }
            if (bestS < 0) break;               // stuck: single trail ends here
            ll ec = (ll)cur * A + bestS;
            used[ec] = 1; cov++;
            S.push_back((char)('0' + bestS));
            cur = (int)(ec % V);
        }
        if (cov > bestCov) { bestCov = cov; best.swap(S); }
    }

    printf("%s\n", best.c_str());
    return 0;
}
