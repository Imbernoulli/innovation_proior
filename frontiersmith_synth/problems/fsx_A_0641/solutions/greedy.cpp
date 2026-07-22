// TIER: greedy
// The obvious first attempt: start from a pseudo-random shuffle, then hill-climb with
// random-pair swaps, accepting a swap iff it does not worsen the worst-channel leak.
// Leak() is recomputed from scratch (O(p^2)) after every candidate swap -- the natural,
// unoptimized way to write a local search. A fixed deterministic operation budget caps the
// iteration count; since each iteration costs O(p^2), the number of iterations collapses at
// large p (an O(1/p^2) iteration budget), so on the largest/densest-weighted tests this
// local search barely moves off its random start, plateauing near a sqrt(p)-scale floor.
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

int main() {
    if (scanf("%d", &P) != 1) return 0;
    W.assign(P, 0);
    for (int d = 1; d <= P - 1; d++) scanf("%d", &W[d]);
    buildChi(P);

    vector<int> sigma(P);
    for (int i = 0; i < P; i++) sigma[i] = i;
    mt19937 rng((unsigned)(P * 2654435761u + 1013904223u));
    shuffle(sigma.begin(), sigma.end(), rng);

    long long cur = leakOf(sigma);

    // Deterministic operation budget: each iteration costs ~P^2 work, so iterations
    // shrink like 1/P^2 at large P -- a realistic naive-implementation slowdown, not an
    // artificial handicap.
    const double OP_BUDGET = 3.0e8;
    long long iters = (long long)(OP_BUDGET / ((double)P * (double)P));
    iters = max(30LL, min(60000LL, iters));

    clock_t startClk = clock();
    const double TIME_CAP = 3.2; // seconds, safety net under the 4s limit
    uniform_int_distribution<int> pick(0, P - 1);
    for (long long it = 0; it < iters; it++) {
        if ((it & 63) == 0) {
            double el = (double)(clock() - startClk) / CLOCKS_PER_SEC;
            if (el > TIME_CAP) break;
        }
        int i = pick(rng), j = pick(rng);
        if (i == j) continue;
        swap(sigma[i], sigma[j]);
        long long nv = leakOf(sigma);
        if (nv <= cur) {
            cur = nv;
        } else {
            swap(sigma[i], sigma[j]);
        }
    }

    for (int i = 0; i < P; i++) printf("%d%c", sigma[i], i + 1 == P ? '\n' : ' ');
    return 0;
}
