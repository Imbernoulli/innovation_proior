// TIER: strong
// Nearest-neighbor construction from several deterministic starts, each
// refined by 2-opt local search on the OPEN Hamiltonian path (segment
// reversal), keeping the best schedule found. Deterministic (fixed seeds).
#include <bits/stdc++.h>
using namespace std;

int n, L;
vector<vector<int>> pat;
vector<int> diffCnt, touched;

int coupling(const vector<int>& A, const vector<int>& B) {
    touched.clear();
    int best = 0;
    for (int a : A) for (int b : B) {
        int idx = a - b + L;
        int v = ++diffCnt[idx];
        if (v == 1) touched.push_back(idx);
        if (v > best) best = v;
    }
    for (int idx : touched) diffCnt[idx] = 0;
    return best;
}

vector<vector<int>> C;

long long pathCost(const vector<int>& o) {
    long long c = 0;
    for (int i = 0; i + 1 < (int)o.size(); i++) c += C[o[i]][o[i + 1]];
    return c;
}

vector<int> nnFrom(int start) {
    vector<char> used(n, 0);
    vector<int> o; o.reserve(n);
    int cur = start; used[cur] = 1; o.push_back(cur);
    for (int step = 1; step < n; step++) {
        int best = -1, bestv = INT_MAX;
        for (int j = 0; j < n; j++)
            if (!used[j] && C[cur][j] < bestv) { bestv = C[cur][j]; best = j; }
        used[best] = 1; o.push_back(best); cur = best;
    }
    return o;
}

// 2-opt on an open path: reverse segment [i..j]; recompute the two boundary
// edges only. Repeat until no improvement (or budget exhausted).
void twoOpt(vector<int>& o, long long budget) {
    bool improved = true;
    long long iters = 0;
    while (improved && iters < budget) {
        improved = false;
        for (int i = 0; i + 1 < n; i++) {
            int a = (i == 0) ? -1 : o[i - 1];
            for (int j = i + 1; j < n; j++) {
                int b = (j + 1 < n) ? o[j + 1] : -1;
                long long before = 0, after = 0;
                if (a >= 0) { before += C[a][o[i]]; after += C[a][o[j]]; }
                if (b >= 0) { before += C[o[j]][b]; after += C[o[i]][b]; }
                if (after < before) {
                    reverse(o.begin() + i, o.begin() + j + 1);
                    improved = true;
                }
                if (++iters >= budget) return;
            }
        }
    }
}

int main() {
    if (scanf("%d %d", &n, &L) != 2) return 0;
    pat.assign(n, {});
    diffCnt.assign(2 * L + 2, 0);
    for (int i = 0; i < n; i++) {
        int s; scanf("%d", &s);
        pat[i].resize(s);
        for (int j = 0; j < s; j++) scanf("%d", &pat[i][j]);
    }
    C.assign(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            int v = coupling(pat[i], pat[j]);
            C[i][j] = C[j][i] = v;
        }

    long long budget = 4000000;               // 2-opt evaluation budget per restart
    // choose a handful of deterministic starts
    vector<int> starts;
    starts.push_back(0);
    starts.push_back(n / 2);
    starts.push_back(n - 1);
    // a couple of pseudo-random starts (fixed seed => deterministic)
    unsigned long long seed = 0x9e3779b97f4a7c15ULL;
    for (int t = 0; t < 3 && (int)starts.size() < n; t++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        starts.push_back((int)(seed % n));
    }

    vector<int> best;
    long long bestCost = LLONG_MAX;
    for (int st : starts) {
        vector<int> o = nnFrom(st);
        twoOpt(o, budget);
        long long c = pathCost(o);
        if (c < bestCost) { bestCost = c; best = o; }
    }

    for (int i = 0; i < n; i++) printf("%d%c", best[i] + 1, i == n - 1 ? '\n' : ' ');
    return 0;
}
