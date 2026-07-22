// TIER: strong
// The insight: chi(x^k) = chi(x)^k, so a monomial permutation sigma(x) = x^k mod p (for k
// coprime to p-1, sigma(0)=0) is a single ALGEBRAIC object whose worst-channel leak can be
// evaluated directly, with no per-test tuning of a combinatorial search -- and empirically
// (Weil-type character-sum cancellation) a handful of such k decorrelate EVERY channel
// simultaneously down near the theoretical floor (|T(d)| = O(1)), regardless of which
// channels the input happens to weight heavily. So instead of locally repairing one
// permutation, we cheaply score a small closed-form family of candidate global maps and
// keep the best -- this needs only ~O(1) candidates, not O(p) local moves.
#include <bits/stdc++.h>
using namespace std;

static int P;
static vector<int> chiTab;
static vector<int> W;

static void buildChi(int p) {
    chiTab.assign(p, -1);
    chiTab[0] = 0;
    for (long long x = 1; x < p; x++) chiTab[(int)((x * x) % p)] = 1;
}

static long long leakOf(const vector<int>& sigma) {
    int p = P;
    long long best = 0;
    for (int d = 1; d <= p - 1; d++) {
        long long T = 0;
        for (int i = 0; i < p; i++) {
            int j = i - d; if (j < 0) j += p;
            int diff = sigma[i] - sigma[j];
            diff %= p; if (diff < 0) diff += p;
            T += chiTab[diff];
        }
        long long val = (long long)W[d] * llabs(T);
        if (val > best) best = val;
    }
    return best;
}

static long long powmod(long long b, long long e, long long m) {
    long long r = 1 % m; b %= m;
    while (e > 0) {
        if (e & 1) r = (r * b) % m;
        b = (b * b) % m;
        e >>= 1;
    }
    return r;
}

static vector<int> monomialPerm(int p, long long a, long long k, long long b) {
    vector<int> s(p);
    s[0] = (int)(((b % p) + p) % p);
    for (int x = 1; x < p; x++) {
        long long v = (a * powmod(x, k, p)) % p;
        v = (v + b) % p; if (v < 0) v += p;
        s[x] = (int)v;
    }
    return s;
}

int main() {
    if (scanf("%d", &P) != 1) return 0;
    W.assign(P, 0);
    for (int d = 1; d <= P - 1; d++) scanf("%d", &W[d]);
    buildChi(P);

    // find a quadratic non-residue (for a second multiplicative "dressing" a) -- any x with
    // chiTab[x] == -1.
    long long nonres = 1;
    for (int x = 1; x < P; x++) if (chiTab[x] == -1) { nonres = x; break; }

    long long bestLeak = LLONG_MAX;
    vector<int> best;

    clock_t startClk = clock();
    const double TIME_CAP = 3.2; // seconds, safety net under the 4s limit
    int tried = 0;
    const int MAX_CANDIDATES = 60;

    for (long long k = 3; k <= 2 * P && tried < MAX_CANDIDATES; k += 2) {
        if (__gcd(k, (long long)(P - 1)) != 1) continue;
        for (long long a : {1LL, nonres}) {
            double el = (double)(clock() - startClk) / CLOCKS_PER_SEC;
            if (el > TIME_CAP) goto done;
            vector<int> s = monomialPerm(P, a, k, 0);
            long long lk = leakOf(s);
            tried++;
            if (lk < bestLeak) { bestLeak = lk; best = s; }
        }
    }
done:
    if (best.empty()) {
        best.resize(P);
        for (int i = 0; i < P; i++) best[i] = i;
    }

    for (int i = 0; i < P; i++) printf("%d%c", best[i], i + 1 == P ? '\n' : ' ');
    return 0;
}
