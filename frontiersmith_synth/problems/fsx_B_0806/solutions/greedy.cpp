// TIER: greedy
// The obvious idea: make the MIRROR LAYOUT symmetric under the target group G, on the
// assumption that symmetric hardware will produce a symmetric light trace. Pick a
// handful of points evenly spaced along the beam's own mirror-free straight path and
// place a mirror at each of them AND at every one of its G-images, alternating '/'
// and '\' for a "balanced" look. In reality only the seeds that sit back on the
// beam's actual route ever get hit -- the very first one deflects the beam away from
// its column almost immediately, and every orbit-image off the true route is inert:
// most of the placed budget never sees the light at all.
#include <bits/stdc++.h>
using namespace std;

static pair<int,int> applyCode(int code, int r, int c, int n) {
    switch (code) {
        case 0: return {r, c};
        case 1: return {c, n - 1 - r};
        case 2: return {n - 1 - r, n - 1 - c};
        case 3: return {n - 1 - c, r};
        case 4: return {n - 1 - r, c};
        case 5: return {r, n - 1 - c};
        case 6: return {c, r};
        default: return {n - 1 - c, n - 1 - r};
    }
}

int main() {
    int n, M, ec;
    char group[8];
    scanf("%d %d %7s", &n, &M, group);
    scanf("%d", &ec);

    vector<int> codes;
    if (string(group) == "C2") codes = {0, 2};
    else codes = {0, 1, 2, 3, 4, 5, 6, 7};
    int g = (int)codes.size();

    int maxSeeds = max(1, M / g);
    int stride = max(1, n / (maxSeeds + 1));

    vector<vector<char>> mirror(n, vector<char>(n, 0));
    int used = 0;
    vector<pair<int,int>> placed;
    for (int k = 0; k < maxSeeds; k++) {
        int idx = min(n - 1, stride * (k + 1));
        int r = idx, c = ec;
        char mtype = (k % 2 == 0) ? '/' : '\\';
        vector<pair<int,int>> cand;
        for (int code : codes) {
            auto p = applyCode(code, r, c, n);
            if (p.first < 0 || p.first >= n || p.second < 0 || p.second >= n) continue;
            if (mirror[p.first][p.second]) continue;
            bool dup = false;
            for (auto& q : cand) if (q == p) { dup = true; break; }
            if (!dup) cand.push_back(p);
        }
        if (used + (int)cand.size() > M) continue;
        for (auto& p : cand) {
            mirror[p.first][p.second] = mtype;
            placed.push_back(p);
            used++;
        }
    }

    printf("%d\n", (int)placed.size());
    for (auto& p : placed) {
        printf("%d %d %c\n", p.first, p.second, mirror[p.first][p.second]);
    }
    return 0;
}
