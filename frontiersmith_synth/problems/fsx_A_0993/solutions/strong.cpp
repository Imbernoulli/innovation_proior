// TIER: strong
// The insight: design in the seam-interface language FIRST. Fix every
// block's border ring (its outer edge cells) to a single constant value --
// this guarantees every seam matches automatically (S = Smax) no matter how
// the interiors vary, because a block's right/bottom edge is then
// identical, cell for cell, to its neighbors' left/top edge. That collapses
// the whole distinctness problem onto the free (b-2)x(b-2) interior, which
// we enumerate with a SINGLE global cursor (never restarted per block) and
// dedupe via the dihedral canonical form (the same 8-symmetry quotient the
// checker uses) -- certifying up to Q pairwise-distinct block patterns as
// cheaply as a hash-set lookup, instead of an O(B^4) pairwise comparison.
// Colors use one GLOBAL continuous round-robin counter (never resets), so
// every color's count stays within O(1) of the even share regardless of B
// -- the balance gate never binds.
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;

// Must match the checker's group action exactly: a physical rotation/
// reflection turns the cell, flipping "\" <-> "/" under 90/270 rotations and
// axis-aligned mirrors (unchanged under 180 rotation and diagonal mirrors).
static u64 rot90(u64 m, int N){
    u64 r = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++){
            int oldbit = (int)((m >> (i * N + j)) & 1ULL);
            int newbit = 1 - oldbit;
            if (newbit) {
                int ni = j, nj = N - 1 - i;
                r |= (1ULL << (ni * N + nj));
            }
        }
    return r;
}
static u64 mirror(u64 m, int N){
    u64 r = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++){
            int oldbit = (int)((m >> (i * N + j)) & 1ULL);
            int newbit = 1 - oldbit;
            if (newbit) {
                int nj = N - 1 - j;
                r |= (1ULL << (i * N + nj));
            }
        }
    return r;
}
static u64 canonForm(u64 m, int N){
    u64 best = m;
    for (int rep = 0; rep < 2; rep++){
        u64 cur = (rep == 0) ? m : mirror(m, N);
        for (int k = 0; k < 4; k++){
            if (cur < best) best = cur;
            cur = rot90(cur, N);
        }
    }
    return best;
}

int main(){
    int B,b,c,Q,tol; long long W1;
    if (scanf("%d %d %d %d %d %lld", &B, &b, &c, &Q, &tol, &W1) != 6) return 0;

    int inN = b - 2; // interior side length (statement guarantees b>=3)
    if (inN < 1) inN = 1; // defensive; not reached under the stated constraints
    long long interiorBits = (long long)inN * inN;
    long long maxRaw = 1LL << interiorBits;

    long long need = min((long long)Q, (long long)B * (long long)B);
    need = min(need, maxRaw);

    vector<u64> chosen;
    set<u64> used;
    for (long long raw = 0; raw < maxRaw && (long long)chosen.size() < need; raw++){
        u64 full = 0;
        for (int i = 0; i < inN; i++)
            for (int j = 0; j < inN; j++)
                if ((raw >> (i * inN + j)) & 1LL)
                    full |= (1ULL << ((i + 1) * b + (j + 1)));
        u64 cf = canonForm(full, b);
        if (used.count(cf)) continue;
        used.insert(cf);
        chosen.push_back((u64)raw);
    }
    if (chosen.empty()) chosen.push_back(0ULL);

    vector<vector<int>> d(B * b, vector<int>(B * b, 0));
    size_t idx = 0;
    for (int br = 0; br < B; br++){
        for (int bc = 0; bc < B; bc++){
            u64 raw = chosen[idx % chosen.size()]; idx++;
            for (int i = 0; i < inN; i++)
                for (int j = 0; j < inN; j++)
                    if ((raw >> (i * inN + j)) & 1ULL)
                        d[br * b + 1 + i][bc * b + 1 + j] = 1;
            // border ring cells stay 0 (already initialized).
        }
    }

    long long K = 0;
    for (int br = 0; br < B; br++)
        for (int bc = 0; bc < B; bc++)
            for (int i = 0; i < b; i++)
                for (int j = 0; j < b; j++){
                    int c1 = (int)(K % c) + 1;
                    int c2 = (int)((K + 1) % c) + 1;
                    printf("%d %d %d\n", d[br * b + i][bc * b + j], c1, c2);
                    K++;
                }
    return 0;
}
