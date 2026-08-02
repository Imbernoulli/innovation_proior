// TIER: strong
// The insight: bids that never share an item can never compete, so the
// winner-determination instance decomposes along the item-sharing graph into
// independent components. Decompose first (union-find over items via each
// bid's item set), THEN act:
//   - a component with few bids is exactly where a fractional/LP-style
//     relaxation would be loose (the hub-bundle-vs-spoiler traps, the
//     weighted-path traps, and the odd conflict rings are all tiny isolated
//     components by construction) -- brute force it EXACTLY instead of
//     trusting a density heuristic there.
//   - a large loose component (the general overlapping background) is cheap
//     to handle with density-greedy plus a bounded local-exchange pass that
//     rescues any bid whose blockers are worth less than it is.
// This recovers the bundle's true value in every hub, the two-disjoint-edges
// answer in every path, and the exact ring matching, none of which a flat
// density-sorted scan (see solutions/greedy.cpp) can see.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int M, N;
vector<ll> cap;
vector<int> bk;
vector<ll> bp;
vector<vector<int>> bitems;

struct DSU {
    vector<int> par, rnk;
    DSU(int n) : par(n + 1), rnk(n + 1, 0) { iota(par.begin(), par.end(), 0); }
    int find(int x) { return par[x] == x ? x : par[x] = find(par[x]); }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rnk[a] < rnk[b]) swap(a, b);
        par[b] = a;
        if (rnk[a] == rnk[b]) rnk[a]++;
    }
};

int main() {
    scanf("%d %d", &M, &N);
    cap.assign(M + 1, 0);
    for (int j = 1; j <= M; j++) scanf("%lld", &cap[j]);
    bk.assign(N + 1, 0); bp.assign(N + 1, 0); bitems.assign(N + 1, {});
    for (int i = 1; i <= N; i++) {
        int k; ll p;
        scanf("%d %lld", &k, &p);
        bk[i] = k; bp[i] = p;
        bitems[i].resize(k);
        for (int t = 0; t < k; t++) scanf("%d", &bitems[i][t]);
    }

    DSU dsu(M);
    for (int i = 1; i <= N; i++)
        for (size_t t = 1; t < bitems[i].size(); t++)
            dsu.unite(bitems[i][0], bitems[i][t]);

    map<int, vector<int>> comps; // root(item) -> bid ids
    for (int i = 1; i <= N; i++) {
        int root = dsu.find(bitems[i][0]);
        comps[root].push_back(i);
    }

    vector<int> accepted;
    const int BF_LIMIT = 18;

    for (auto &kv : comps) {
        vector<int> &cb = kv.second;
        int b = (int)cb.size();
        if (b <= BF_LIMIT) {
            // ---- exact brute force over all 2^b subsets ----
            // local item list for fast capacity bookkeeping
            vector<int> localItems;
            for (int idx : cb) for (int it : bitems[idx]) localItems.push_back(it);
            sort(localItems.begin(), localItems.end());
            localItems.erase(unique(localItems.begin(), localItems.end()), localItems.end());
            int L = (int)localItems.size();
            unordered_map<int, int> pos;
            for (int t = 0; t < L; t++) pos[localItems[t]] = t;

            ll bestVal = -1;
            int bestMask = 0;
            vector<int> use(L);
            for (int mask = 0; mask < (1 << b); mask++) {
                fill(use.begin(), use.end(), 0);
                bool ok = true;
                ll val = 0;
                for (int t = 0; t < b && ok; t++) {
                    if (!(mask & (1 << t))) continue;
                    int idx = cb[t];
                    for (int it : bitems[idx]) {
                        int p2 = pos[it];
                        use[p2]++;
                        if (use[p2] > cap[it]) { ok = false; break; }
                    }
                    if (ok) val += bp[idx];
                }
                if (ok && val > bestVal) { bestVal = val; bestMask = mask; }
            }
            for (int t = 0; t < b; t++) if (bestMask & (1 << t)) accepted.push_back(cb[t]);
        } else {
            // ---- large loose component: density-greedy + local exchange ----
            vector<int> order = cb;
            sort(order.begin(), order.end(), [&](int a, int c) {
                __int128 lhs = (__int128)bp[a] * bk[c];
                __int128 rhs = (__int128)bp[c] * bk[a];
                if (lhs != rhs) return lhs > rhs;
                if (bp[a] != bp[c]) return bp[a] > bp[c];
                return a < c;
            });

            vector<ll> rem = cap;
            vector<char> isAcc(N + 1, 0);
            map<int, vector<int>> owners; // item -> accepted bid ids using it
            auto doAccept = [&](int idx) {
                isAcc[idx] = 1;
                for (int it : bitems[idx]) { rem[it]--; owners[it].push_back(idx); }
            };
            auto doEvict = [&](int idx) {
                isAcc[idx] = 0;
                for (int it : bitems[idx]) {
                    rem[it]++;
                    auto &v = owners[it];
                    v.erase(find(v.begin(), v.end(), idx));
                }
            };

            for (int idx : order) {
                bool ok = true;
                for (int it : bitems[idx]) if (rem[it] < 1) { ok = false; break; }
                if (ok) doAccept(idx);
            }

            // rescue pass: try to admit unaccepted bids, highest price first,
            // evicting the cheapest blockers if that is a net improvement.
            vector<int> byPrice = cb;
            sort(byPrice.begin(), byPrice.end(), [&](int a, int c) { return bp[a] > bp[c]; });
            for (int pass = 0; pass < 2; pass++) {
                for (int idx : byPrice) {
                    if (isAcc[idx]) continue;
                    set<int> evictSet;
                    bool feasible = true;
                    for (int it : bitems[idx]) {
                        ll needed = rem[it] < 1 ? 1 - rem[it] : 0;
                        // count how many owners of `it` are not yet marked for eviction
                        vector<int> cand;
                        for (int o : owners[it]) if (!evictSet.count(o)) cand.push_back(o);
                        // evict cheapest-first until this item's deficit is covered
                        sort(cand.begin(), cand.end(), [&](int x, int y) { return bp[x] < bp[y]; });
                        ll have = rem[it];
                        for (int o : cand) {
                            if (have >= 1) break;
                            evictSet.insert(o);
                            have++;
                        }
                        if (have < 1) { feasible = false; break; }
                    }
                    if (!feasible) continue;
                    ll evictTotal = 0;
                    for (int o : evictSet) evictTotal += bp[o];
                    if (evictTotal < bp[idx]) {
                        for (int o : evictSet) doEvict(o);
                        doAccept(idx);
                    }
                }
            }

            for (int idx : cb) if (isAcc[idx]) accepted.push_back(idx);
        }
    }

    printf("%d\n", (int)accepted.size());
    for (size_t i = 0; i < accepted.size(); i++)
        printf("%d%c", accepted[i], i + 1 == accepted.size() ? '\n' : ' ');
    if (accepted.empty()) printf("\n");
    return 0;
}
