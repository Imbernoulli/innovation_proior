#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;
typedef __int128 i128;

static int n;
static vector<long long> a;

// Two independent 61-bit Mersenne-prime hashes to make collisions astronomically rare.
static const u64 MOD = (1ULL << 61) - 1;

static inline u64 mulmod(u64 x, u64 y) {
    i128 z = (i128)x * y;
    u64 lo = (u64)(z & MOD);
    u64 hi = (u64)(z >> 61);
    u64 r = lo + hi;
    if (r >= MOD) r -= MOD;
    return r;
}
static inline u64 addmod(u64 x, u64 y) {
    u64 r = x + y;
    if (r >= MOD) r -= MOD;
    return r;
}

static u64 BASE1, BASE2;
static vector<u64> pre1, pre2;   // prefix hashes
static vector<u64> pw1, pw2;     // base powers

// hashed symbol for a value: offset so negatives/zeros become positive and never 0.
// |a[i]| <= 1e9, so a[i] + OFFSET is in [1, ~2e9], strictly positive => no symbol hashes to 0.
static const u64 OFFSET = 1000000001ULL;

static inline u64 sym(long long v) {
    return (u64)(v + (long long)OFFSET); // in [1, 2000000001], always > 0
}

// hash of subarray a[l .. l+L-1] (0-indexed), combined 128-bit-ish key.
static inline pair<u64,u64> windowHash(int l, int L) {
    int r = l + L; // exclusive
    // h = pre[r] - pre[l]*pw[L]
    u64 h1 = (pre1[r] + (MOD - mulmod(pre1[l], pw1[L]))) % MOD;
    u64 h2 = (pre2[r] + (MOD - mulmod(pre2[l], pw2[L]))) % MOD;
    return {h1, h2};
}

static inline bool windowEqual(int i, int j, int L) {
    for (int k = 0; k < L; k++) if (a[i + k] != a[j + k]) return false;
    return true;
}

// Does some block of length L occur at >=2 distinct start positions (overlap allowed)?
static bool hasDup(int L) {
    if (L <= 0) return true;          // empty block trivially "repeats"; not queried for L>=1
    if (L > n) return false;
    // map combined hash key -> list of start indices sharing that key; verify on clash.
    unordered_map<u64, vector<int>> seen;
    seen.reserve((size_t)(n - L + 1) * 2 + 4);
    for (int i = 0; i + L <= n; i++) {
        auto h = windowHash(i, L);
        u64 key = h.first * 1000000007ULL + h.second; // mix two 61-bit hashes
        auto &bucket = seen[key];
        for (int j : bucket) if (windowEqual(j, i, L)) return true; // exact confirmation
        bucket.push_back(i);
    }
    return false;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n)) { cout << 0 << "\n"; return 0; }
    a.resize(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // random bases in a safe range [256, MOD-2]
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    BASE1 = 256 + rng() % (MOD - 300);
    BASE2 = 256 + rng() % (MOD - 300);

    pre1.assign(n + 1, 0);
    pre2.assign(n + 1, 0);
    pw1.assign(n + 1, 1);
    pw2.assign(n + 1, 1);
    for (int i = 0; i < n; i++) {
        pre1[i + 1] = addmod(mulmod(pre1[i], BASE1), sym(a[i]) % MOD);
        pre2[i + 1] = addmod(mulmod(pre2[i], BASE2), sym(a[i]) % MOD);
        pw1[i + 1] = mulmod(pw1[i], BASE1);
        pw2[i + 1] = mulmod(pw2[i], BASE2);
    }

    // Binary search the largest L in [1, n] with hasDup(L) true.
    // Monotone: if a block of length L repeats, its length-(L-1) prefixes also repeat.
    int lo = 1, hi = n, ans = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (hasDup(mid)) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }

    cout << ans << "\n";
    return 0;
}
