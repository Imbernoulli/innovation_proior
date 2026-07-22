// TIER: strong
// Insight: SEED1/SEED2 and the whole trace are fully known, so instead of
// trusting the closed-form average-false-positive-rate formula, EXACTLY
// simulate candidate allocations (build the real Bloom filter, replay the
// real trace, read off the real ink cost) and use that as the search
// objective. Start every checkpoint at the bit floor, then repeatedly hand
// a budget chunk to whichever checkpoint's REAL simulated total cost drops
// the most (cost-weighted by realized cascade impact, not by watch-list
// size), and finally refine each k_i against the exact replay. This
// naturally overspends on sparsely-populated deep checkpoints (their false
// alarms are the expensive ones) instead of equalizing an analytic rate.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

static inline u64 mix(u64 x, u64 seed) {
    x += seed + 0x9e3779b97f4a7c15ULL;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

int T;
ll B;
u64 SEED1, SEED2;
vector<vector<ll>> C;
unordered_map<ll,int> trueTierOf;
vector<ll> trace;

vector<ll> m; vector<int> k; vector<vector<uint8_t>> filt;

static vector<uint8_t> buildFilter(int i, ll mm, int kk) {
    vector<uint8_t> bits((size_t)mm, 0);
    for (ll x : C[i]) {
        u64 h1 = mix((u64)x, SEED1), h2 = mix((u64)x, SEED2);
        for (int j = 0; j < kk; j++) {
            u64 pos = (h1 + (u64)j * h2) % (u64)mm;
            bits[pos] = 1;
        }
    }
    return bits;
}

static inline bool present(const vector<uint8_t>& bits, ll mm, int kk, ll x) {
    u64 h1 = mix((u64)x, SEED1), h2 = mix((u64)x, SEED2);
    for (int j = 0; j < kk; j++) {
        u64 pos = (h1 + (u64)j * h2) % (u64)mm;
        if (!bits[pos]) return false;
    }
    return true;
}

static ll replay() {
    ll total = 0;
    for (ll x : trace) {
        auto it = trueTierOf.find(x);
        int trueTier = (it == trueTierOf.end()) ? -1 : it->second;
        for (int i = 1; i <= T; i++) {
            if (present(filt[i], m[i], k[i], x)) {
                total += (ll)1 << (i - 1);
                if (trueTier == i) break;
            }
        }
    }
    return total;
}

int main() {
    ios::sync_with_stdio(false); cin.tie(0);
    cin >> T >> B >> SEED1 >> SEED2;
    C.assign(T + 1, {});
    vector<int> n(T + 1, 0);
    for (int i = 1; i <= T; i++) {
        int ni; cin >> ni; n[i] = ni;
        C[i].resize(ni);
        for (int j = 0; j < ni; j++) { cin >> C[i][j]; trueTierOf[C[i][j]] = i; }
    }
    ll Q; cin >> Q;
    trace.resize(Q);
    for (ll q = 0; q < Q; q++) cin >> trace[q];

    const ll M_MIN = 64; const int K_MAX = 10;
    m.assign(T + 1, M_MIN);
    k.assign(T + 1, 4);
    filt.assign(T + 1, {});
    for (int i = 1; i <= T; i++) filt[i] = buildFilter(i, m[i], k[i]);

    ll leftover = B - (ll)T * M_MIN;
    if (leftover < 0) leftover = 0;

    const int STEPS = 10;
    ll chunk = max(1LL, leftover / STEPS);
    ll remaining = leftover;

    // ---- phase 1: cost-weighted bit-budget apportionment via exact replay ----
    while (remaining > 0) {
        ll step = min(chunk, remaining);
        int bestI = -1; ll bestCost = LLONG_MAX;
        vector<uint8_t> bestFilt;
        for (int i = 1; i <= T; i++) {
            ll savedM = m[i];
            vector<uint8_t> trial = buildFilter(i, savedM + step, k[i]);
            filt[i].swap(trial);
            m[i] = savedM + step;
            ll cost = replay();
            if (cost < bestCost) { bestCost = cost; bestI = i; bestFilt = filt[i]; }
            filt[i].swap(trial);
            m[i] = savedM;
        }
        m[bestI] += step;
        filt[bestI] = std::move(bestFilt);
        remaining -= step;
    }

    // ---- phase 2: exact-replay k refinement (dodge the specific colliding ids) ----
    for (int pass = 0; pass < 1; pass++) {
        for (int i = 1; i <= T; i++) {
            int bestK = k[i]; ll bestCost = LLONG_MAX;
            vector<uint8_t> bestFilt;
            for (int kk = 1; kk <= K_MAX; kk++) {
                vector<uint8_t> trial = buildFilter(i, m[i], kk);
                int savedK = k[i];
                filt[i].swap(trial);
                k[i] = kk;
                ll cost = replay();
                if (cost < bestCost) { bestCost = cost; bestK = kk; bestFilt = filt[i]; }
                filt[i].swap(trial);
                k[i] = savedK;
            }
            k[i] = bestK; filt[i] = std::move(bestFilt);
        }
    }

    for (int i = 1; i <= T; i++) cout << m[i] << " " << k[i] << "\n";
    return 0;
}
