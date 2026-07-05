// TIER: strong
#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<int> pk;
vector<vector<pair<int,int>>> pat;
vector<long long> pw;
vector<vector<pair<int,int>>> stlist; // station -> list of (pattern, required orientation)

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    pk.resize(m); pat.resize(m); pw.resize(m);
    stlist.assign(n + 1, {});
    for (int j = 0; j < m; j++) {
        int k; scanf("%d", &k);
        pk[j] = k; pat[j].resize(k);
        for (int e = 0; e < k; e++) {
            int s, o; scanf("%d %d", &s, &o);
            pat[j][e] = {s, o};
            stlist[s].push_back({j, o});
        }
        long long w; scanf("%lld", &w); pw[j] = w;
    }

    // greedy start (same construction as greedy.cpp)
    vector<int> greedyB(n + 1, 0);
    {
        vector<int> assign(n + 1, -1);
        vector<int> idx(m);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b){ return pw[a] > pw[b]; });
        for (int id : idx) {
            bool f = true;
            for (auto& pr : pat[id])
                if (assign[pr.first] != -1 && assign[pr.first] != pr.second) { f = false; break; }
            if (f) for (auto& pr : pat[id]) assign[pr.first] = pr.second;
        }
        for (int i = 1; i <= n; i++) greedyB[i] = assign[i] == -1 ? 0 : assign[i];
    }

    auto matchedInit = [&](const vector<int>& b, vector<int>& matched) -> long long {
        matched.assign(m, 0);
        for (int j = 0; j < m; j++) {
            int c = 0;
            for (auto& pr : pat[j]) if (b[pr.first] == pr.second) c++;
            matched[j] = c;
        }
        long long F = 0;
        for (int j = 0; j < m; j++) if (matched[j] == pk[j]) F += pw[j];
        return F;
    };

    auto hill = [&](vector<int> b) -> pair<long long, vector<int>> {
        vector<int> matched;
        long long F = matchedInit(b, matched);
        bool improved = true; int sweeps = 0;
        while (improved && sweeps < 40) {
            improved = false; sweeps++;
            for (int i = 1; i <= n; i++) {
                long long delta = 0;
                for (auto& pr : stlist[i]) {
                    int p = pr.first, req = pr.second;
                    bool cur = (b[i] == req);
                    if (cur) { if (matched[p] == pk[p]) delta -= pw[p]; }
                    else     { if (matched[p] == pk[p] - 1) delta += pw[p]; }
                }
                if (delta > 0) {
                    b[i] ^= 1; F += delta;
                    for (auto& pr : stlist[i]) {
                        int p = pr.first, req = pr.second;
                        if (b[i] == req) matched[p]++; else matched[p]--;
                    }
                    improved = true;
                }
            }
        }
        return {F, b};
    };

    long long bestF = -1;
    vector<int> bestB(n + 1, 0);

    vector<vector<int>> starts;
    starts.push_back(greedyB);
    starts.push_back(vector<int>(n + 1, 0));
    mt19937 rng(0xC0FFEE);
    for (int r = 0; r < 5; r++) {
        vector<int> rb(n + 1, 0);
        for (int i = 1; i <= n; i++) rb[i] = rng() & 1u;
        starts.push_back(rb);
    }

    for (auto& s : starts) {
        auto res = hill(s);
        if (res.first > bestF) { bestF = res.first; bestB = res.second; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", bestB[i], i == n ? '\n' : ' ');
    if (n == 0) printf("\n");
    return 0;
}
