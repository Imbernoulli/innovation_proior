// TIER: strong
// Insight: co-design the three levers instead of packing a uniform B-tree.
//   (1) Depth-bias: build by WEIGHTED-quantile splits so hot tiles become shallow
//       separators (an alphabetic/Huffman-ish tree, order-constrained).
//   (2) Contiguity: SUPPRESS the split-weight of tiles that are heavily SCANNED, so a
//       swept band is NOT shattered into singletons -- it collapses into few fat pages
//       that a scan sweeps in one touch. (Read the scans, not just the demands.)
//   (3) Pack leaves of <= B tiles (line-count 1). An even-split fallback keeps the tree
//       balanced (never worse than the uniform recipe on cold regions).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, B, S;
vector<ll> ew;          // effective split weight (1-indexed)
vector<ll> PW;          // prefix of (ew+1)
vector<vector<int>> nodeKeys, nodeCh;
int cnt = 0;

int newNode(){ nodeKeys.emplace_back(); nodeCh.emplace_back(); return ++cnt; }

// smallest key t in [lo,hi] with PW[t]-PW[lo-1] >= target
int findSep(int lo, int hi, ll target){
    int a = lo, b = hi, res = hi;
    ll base = PW[lo - 1];
    while (a <= b){
        int mid = (a + b) / 2;
        if (PW[mid] - base >= target){ res = mid; b = mid - 1; }
        else a = mid + 1;
    }
    return res;
}

vector<int> evenSeps(int lo, int hi){
    int size = hi - lo + 1;
    int rem = size - B, basec = rem / (B + 1), extra = rem % (B + 1), cur = lo;
    vector<int> ks;
    for (int j = 0; j <= B; j++){
        int cc = basec + (j < extra ? 1 : 0);
        cur += cc;
        if (j < B){ ks.push_back(cur); cur++; }
    }
    return ks;
}

int maxChildSize(int lo, int hi, const vector<int>& seps){
    int mx = 0, prev = lo;
    for (int s : seps){ mx = max(mx, s - prev); prev = s + 1; }
    mx = max(mx, hi - prev + 1);
    return mx;
}

int build(int lo, int hi){
    if (lo > hi) return 0;
    int size = hi - lo + 1;
    int id = newNode();
    if (size <= B){
        vector<int> ks; for (int k = lo; k <= hi; k++) ks.push_back(k);
        nodeKeys[id - 1] = ks;
        nodeCh[id - 1].assign((int)ks.size() + 1, 0);
        return id;
    }
    // weighted-quantile separators on effective weight
    ll totalW = PW[hi] - PW[lo - 1];
    vector<int> seps;
    int prev = 0;
    for (int j = 1; j <= B; j++){
        ll target = (ll)((__int128)totalW * j / (B + 1));
        if (target < 1) target = 1;
        int s = findSep(lo, hi, target);
        if (s < lo) s = lo; if (s > hi) s = hi;
        if (s != prev){ seps.push_back(s); prev = s; }
    }
    // ensure strictly increasing & distinct (findSep is monotone so already sorted)
    // decide progress; fall back to a balanced even split if weighting stalls
    bool ok = !seps.empty();
    if (ok){
        int mc = maxChildSize(lo, hi, seps);
        if (mc >= size || mc > (int)(0.85 * size)) ok = false;
    }
    if (!ok) seps = evenSeps(lo, hi);

    // build children between separators
    vector<int> chIds;
    int p = lo;
    for (int s : seps){ chIds.push_back(build(p, s - 1)); p = s + 1; }
    chIds.push_back(build(p, hi));
    nodeKeys[id - 1] = seps;
    nodeCh[id - 1] = chIds;
    return id;
}

int main(){
    scanf("%d %d %d", &N, &B, &S);
    vector<ll> w(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%lld", &w[i]);
    // scan coverage via difference array
    vector<ll> cov(N + 2, 0);
    for (int i = 0; i < S; i++){
        ll lo, hi, c; scanf("%lld %lld %lld", &lo, &hi, &c);
        cov[lo] += c; cov[hi + 1] -= c;
    }
    for (int i = 1; i <= N; i++) cov[i] += cov[i - 1];

    // effective split weight: suppress heavily-scanned tiles (keep swept bands packed)
    const ll TAU = 40;   // scan-coverage above which demand is discounted
    ew.assign(N + 1, 0);
    for (int i = 1; i <= N; i++){
        if (cov[i] <= TAU) ew[i] = w[i];
        else ew[i] = max((ll)1, w[i] / (1 + cov[i] / TAU));
    }
    PW.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) PW[i] = PW[i - 1] + ew[i] + 1;

    nodeKeys.reserve(2 * (N / max(1, B) + 8));
    nodeCh.reserve(2 * (N / max(1, B) + 8));
    int root = build(1, N);

    printf("%d %d\n", cnt, root);
    for (int id = 1; id <= cnt; id++){
        auto &ks = nodeKeys[id - 1];
        auto &cs = nodeCh[id - 1];
        printf("%d", (int)ks.size());
        for (int k : ks) printf(" %d", k);
        for (int c : cs) printf(" %d", c);
        printf("\n");
    }
    return 0;
}
