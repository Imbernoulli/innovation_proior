#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

// -----------------------------------------------------------------------------
// Checker / scorer for "Nested Customs Stamps".
//
// Input:  T B SEED1 SEED2 ; then for i=1..T: n_i and n_i watch-list IDs ; then
//         Q and Q trace IDs.
// Output: T lines "m_i k_i" -- ledger size (bits) and stamp count (hashes) for
//         checkpoint i, subject to m_i>=64, 1<=k_i<=10, sum(m_i)<=B.
//
// The checker builds every checkpoint's real Bloom filter bit array from
// SEED1/SEED2 and the given watch-list, then replays the trace container by
// container (checkpoint order 1..T, "absent" = free pass to next checkpoint,
// "present" = pay 2^(i-1) ink for a real check; true member resolves and
// stops, false alarm keeps going). F = total ink. Baseline B_ref = same
// replay under an equal-split m_i=floor(B/T), fixed k=1 reference allocation.
// Score (min): sc = min(1000, 100*B_ref/max(1,F)).
// -----------------------------------------------------------------------------

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
vector<vector<ll>> C;      // C[i] = watch-list of checkpoint i, i=1..T
vector<int> nOf;           // nOf[i] = |C[i]|
unordered_map<ll,int> trueTierOf; // container id -> tier it truly belongs to (1..T)
ll Q;
vector<ll> trace;

// Build checkpoint i's Bloom filter bit array (0/1 bytes) of size m, k stamps.
static vector<uint8_t> buildFilter(int i, ll m, int k) {
    vector<uint8_t> bits((size_t)m, 0);
    for (ll x : C[i]) {
        u64 h1 = mix((u64)x, SEED1), h2 = mix((u64)x, SEED2);
        for (int j = 0; j < k; j++) {
            u64 pos = (h1 + (u64)j * h2) % (u64)m;
            bits[pos] = 1;
        }
    }
    return bits;
}

static inline bool present(const vector<uint8_t>& bits, ll m, int k, ll x) {
    u64 h1 = mix((u64)x, SEED1), h2 = mix((u64)x, SEED2);
    for (int j = 0; j < k; j++) {
        u64 pos = (h1 + (u64)j * h2) % (u64)m;
        if (!bits[pos]) return false;
    }
    return true;
}

// Replay the whole trace against a full allocation; return total ink cost.
static ll replay(const vector<ll>& mArr, const vector<int>& kArr, const vector<vector<uint8_t>>& filt) {
    ll total = 0;
    for (ll x : trace) {
        auto it = trueTierOf.find(x);
        int trueTier = (it == trueTierOf.end()) ? -1 : it->second;
        for (int i = 1; i <= T; i++) {
            if (present(filt[i], mArr[i], kArr[i], x)) {
                total += (ll)1 << (i - 1);
                if (trueTier == i) break; // resolved
            }
        }
    }
    return total;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    // T is fixed at 8 by the statement/generator; B, n_i, sum n_i and Q are
    // parsed against the exact envelope statement.txt promises (T=8 fixed,
    // n_i<=20000, sum n_i<=40000, Q<=40000, B<=2*10^6). Any trusted-input
    // violation here is a generator bug (_fail), never a participant issue.
    T = inf.readInt(8, 8, "T");
    B = inf.readLong(1, (ll)2000000, "B");
    SEED1 = (u64)inf.readLong(1, (ll)5e18, "SEED1");
    SEED2 = (u64)inf.readLong(1, (ll)5e18, "SEED2");
    if (B < (ll)T * 64) quitf(_fail, "generator bug: B=%lld too small for T=%d ledgers of >=64 bits", B, T);

    C.assign(T + 1, {});
    nOf.assign(T + 1, 0);
    ll sumN = 0;
    for (int i = 1; i <= T; i++) {
        int ni = inf.readInt(1, 20000, "n_i");
        nOf[i] = ni;
        sumN += ni;
        C[i].resize(ni);
        for (int j = 0; j < ni; j++) {
            ll x = inf.readLong(1, (ll)2e9, "watch_id");
            C[i][j] = x;
            if (trueTierOf.count(x)) quitf(_fail, "generator bug: duplicate watch-list id %lld", x);
            trueTierOf[x] = i;
        }
    }
    if (sumN > 40000) quitf(_fail, "generator bug: sum n_i=%lld > 40000", sumN);
    Q = inf.readLong(1, (ll)40000, "Q");
    trace.resize(Q);
    for (ll q = 0; q < Q; q++) trace[q] = inf.readLong(1, (ll)2e9, "trace_id");

    // ---- internal baseline: equal split, fixed k=1 ----
    vector<ll> mBase(T + 1); vector<int> kBase(T + 1);
    ll baseM = B / T;
    for (int i = 1; i <= T; i++) { mBase[i] = max(1LL, baseM); kBase[i] = 1; }
    vector<vector<uint8_t>> filtBase(T + 1);
    for (int i = 1; i <= T; i++) filtBase[i] = buildFilter(i, mBase[i], kBase[i]);
    ll Bref = replay(mBase, kBase, filtBase);
    if (Bref <= 0) Bref = 1;

    // ---- read participant output ----
    vector<ll> m(T + 1); vector<int> k(T + 1);
    ll sumM = 0;
    for (int i = 1; i <= T; i++) {
        ll mi = ouf.readLong(64, (ll)4e6, "m_i");
        int ki = ouf.readInt(1, 10, "k_i");
        m[i] = mi; k[i] = ki;
        sumM += mi;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");
    if (sumM > B) quitf(_wa, "bit budget exceeded: sum m_i=%lld > B=%lld", sumM, B);

    vector<vector<uint8_t>> filt(T + 1);
    for (int i = 1; i <= T; i++) filt[i] = buildFilter(i, m[i], k[i]);
    ll F = replay(m, k, filt);

    double sc = min(1000.0, 100.0 * (double)Bref / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, Bref, sc / 1000.0);
    return 0;
}
