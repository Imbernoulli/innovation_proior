#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

// -----------------------------------------------------------------------------
// Checker / scorer for "Combination-Lock Rewiring Against Nudge Attacks".
// family: flat-differential-permutation
//
// Input:  one odd prime p.
// Output: p integers pi(0..p-1): a claimed rewiring (permutation of 0..p-1).
//
// Feasibility: pi must be a genuine permutation of {0,...,p-1} AND a derangement
//   (pi(x) != x for every x). Any violation -> score 0.
//
// Objective (MIN): F = peak leakage = max over shift s in {1,...,p-1} and change
//   c in {0,...,p-1} of count(s,c), where count(s,c) = #{x : pi((x+s) mod p) -
//   pi(x) mod p == c}.
//
// Baseline B (checker-computed, "no attack-resistance logic" reference): split
//   {0,...,p-1} into consecutive blocks of size round(5*sqrt(p)); independently
//   Fisher-Yates shuffle each block using a fixed splitmix64 stream seeded from p;
//   repair leftover fixed points by swapping each with its right neighbour
//   (repeated passes until none remain). B = that construction's own peak leakage.
//   This EXACT procedure is what solutions/trivial.cpp reproduces (-> ratio 0.1).
//
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static u64 rngstate;
static u64 next_rand(){
    rngstate += 0x9E3779B97F4A7C15ULL;
    u64 z = rngstate;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return z;
}

// Reference baseline construction (must match solutions/trivial.cpp exactly).
static vector<int> reference_rewiring(int p){
    int bs = max(2, (int)llround(5.0 * sqrt((double)p)));
    vector<int> pi(p);
    for (int i = 0; i < p; i++) pi[i] = i;
    rngstate = (u64)p * 0x2545F4914F6CDD1DULL + 777ULL;
    for (int start = 0; start < p; start += bs){
        int end = min(p, start + bs);
        for (int i = end - 1; i > start; i--){
            int j = start + (int)(next_rand() % (u64)(i - start + 1));
            swap(pi[i], pi[j]);
        }
    }
    for (int pass = 0; pass < 12; pass++){
        bool changed = false;
        for (int x = 0; x < p; x++){
            if (pi[x] == x){
                int y = (x + 1) % p;
                swap(pi[x], pi[y]);
                changed = true;
            }
        }
        if (!changed) break;
    }
    return pi;
}

// Peak leakage: max over shifts s=1..p-1 of the max bucket count of
// (pi[(x+s) mod p] - pi[x]) mod p over x=0..p-1.  O(p^2), fine for p <= ~7351.
static int peak_leakage(const vector<int>& pi, int p){
    vector<int> cnt(p, 0);
    vector<int> pi2(2 * p);
    for (int i = 0; i < p; i++){ pi2[i] = pi[i]; pi2[i + p] = pi[i]; }
    int best = 0;
    for (int s = 1; s < p; s++){
        fill(cnt.begin(), cnt.end(), 0);
        for (int x = 0; x < p; x++){
            int d = pi2[x + s] - pi2[x];
            if (d < 0) d += p;
            int c = ++cnt[d];
            if (c > best) best = c;
        }
    }
    return best;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int p = inf.readInt();

    vector<int> pi(p);
    vector<char> seen(p, 0);
    for (int x = 0; x < p; x++){
        int v = ouf.readInt(0, p - 1, "pi_x");
        if (seen[v]) quitf(_wa, "value %d output more than once (not a permutation)", v);
        seen[v] = 1;
        pi[x] = v;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the permutation");
    for (int x = 0; x < p; x++){
        if (pi[x] == x) quitf(_wa, "position %d is left unrewired (pi(x) = x)", x);
    }

    int F = peak_leakage(pi, p);

    auto refpi = reference_rewiring(p);
    int B = peak_leakage(refpi, p);
    if (B <= 0) B = 1;   // defensive; construction always yields B>=2 in practice

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1, F));
    quitp(sc / 1000.0, "OK F=%d B=%d Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
