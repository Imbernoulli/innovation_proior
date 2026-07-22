// TIER: strong
// Insight: the T-step update is linear over GF(2), so it equals a fixed NxN matrix
// A obtained from the single-step matrix S by binary exponentiation (S^T). Solving
// A*x = target by Gauss-Jordan elimination gives ONE particular precursor plus a
// NULL-SPACE basis (seed changes the T-step map cannot see -- the rule's invariant
// directions). A single elimination order rarely lands on a low-weight solution
// when the null space is large, so this repeats Gauss-Jordan with several RANDOM
// pivot-column orders (information-set-decoding style) -- each order yields a
// different coset representative -- refines every candidate with local coordinate
// descent over the null space, and keeps the lowest-weight one that fits budget B.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int MB = 256;
typedef bitset<MB> BM;

int N, T, B, M;

vector<BM> matmul(const vector<BM>& X, const vector<BM>& Y, int n){
    vector<BM> Z(n);
    for (int i = 0; i < n; i++){
        BM zi;
        for (int j = 0; j < n; j++) if (X[i][j]) zi ^= Y[j];
        Z[i] = zi;
    }
    return Z;
}

int weightOf(const BM& x, int n){ int c = 0; for (int i = 0; i < n; i++) if (x[i]) c++; return c; }

// Gauss-Jordan solving A x = target (mod 2), pivoting columns in the order given by
// 'colOrder' rather than 0..N-1. Returns (consistent, x0, nullBasis).
bool eliminate(const vector<BM>& A, const vector<char>& rhsIn, int n, const vector<int>& colOrder,
               BM& x0, vector<BM>& nulls){
    vector<BM> coef = A;
    vector<char> rhs = rhsIn;
    vector<int> pivotRowOf(n, -1);
    int pr = 0;
    for (int idx = 0; idx < n && pr < n; idx++){
        int c = colOrder[idx];
        int r = -1;
        for (int rr = pr; rr < n; rr++) if (coef[rr][c]) { r = rr; break; }
        if (r < 0) continue;
        swap(coef[pr], coef[r]); swap(rhs[pr], rhs[r]);
        for (int rr = 0; rr < n; rr++)
            if (rr != pr && coef[rr][c]){ coef[rr] ^= coef[pr]; rhs[rr] ^= rhs[pr]; }
        pivotRowOf[c] = pr;
        pr++;
    }
    for (int rr = pr; rr < n; rr++) if (rhs[rr]) return false;   // inconsistent

    vector<int> freeCols;
    for (int c = 0; c < n; c++) if (pivotRowOf[c] < 0) freeCols.push_back(c);

    x0 = BM();
    for (int c = 0; c < n; c++) if (pivotRowOf[c] >= 0) x0[c] = rhs[pivotRowOf[c]];

    nulls.clear();
    for (int f : freeCols){
        BM nv; nv[f] = 1;
        for (int c = 0; c < n; c++) if (pivotRowOf[c] >= 0) nv[c] = coef[pivotRowOf[c]][f];
        nulls.push_back(nv);
    }
    return true;
}

BM coordDescent(BM cand, const vector<BM>& nulls, int n){
    for (int pass = 0; pass < 8; pass++){
        bool improved = false;
        for (auto& nv : nulls){
            BM alt = cand ^ nv;
            if (weightOf(alt, n) < weightOf(cand, n)){ cand = alt; improved = true; }
        }
        if (!improved) break;
    }
    return cand;
}

int main(){
    scanf("%d %d %d %d", &N, &T, &B, &M);
    char buf[300];
    scanf("%299s", buf);
    string target(buf);

    // ---- single-step matrix S: S[i][(i+d)%N] = 1 for offsets d selected by M ----
    vector<BM> S(N);
    for (int i = 0; i < N; i++){
        for (int d = -2; d <= 2; d++){
            if (M & (1 << (d + 2))){
                int j = ((i + d) % N + N) % N;
                S[i][j] = 1;
            }
        }
    }
    // ---- A = S^T via binary exponentiation (all factors are powers of S -> commute) ----
    vector<BM> A(N); for (int i = 0; i < N; i++) A[i][i] = 1;   // identity
    vector<BM> base = S;
    int e = T;
    while (e > 0){
        if (e & 1) A = matmul(A, base, N);
        base = matmul(base, base, N);
        e >>= 1;
    }
    vector<char> rhs(N);
    for (int i = 0; i < N; i++) rhs[i] = (target[i] == '1') ? 1 : 0;

    mt19937 rngGen(12345u + N * 977u + T * 131u + M);
    vector<int> colOrder(N); for (int i = 0; i < N; i++) colOrder[i] = i;

    BM best; int bestW = INT_MAX; bool haveAny = false;
    int TRIALS = 120;
    for (int trial = 0; trial < TRIALS; trial++){
        if (trial > 0) for (int i = N - 1; i > 0; i--) swap(colOrder[i], colOrder[(int)(rngGen() % (i + 1))]);
        BM x0; vector<BM> nulls;
        if (!eliminate(A, rhs, N, colOrder, x0, nulls)) continue;   // inconsistent (shouldn't happen)
        haveAny = true;
        BM cand = coordDescent(x0, nulls, N);
        int w = weightOf(cand, N);
        if (w < bestW){ best = cand; bestW = w; }
        // a couple of randomized-restart refinements within this pivoting too
        for (int r = 0; r < 3 && !nulls.empty(); r++){
            BM c2 = x0;
            for (auto& nv : nulls) if (rngGen() & 1) c2 ^= nv;
            c2 = coordDescent(c2, nulls, N);
            int w2 = weightOf(c2, N);
            if (w2 < bestW){ best = c2; bestW = w2; }
        }
        if (bestW <= max(1, B / 2)) break;   // already excellent, stop early
    }

    vector<int> chosen;
    if (haveAny && bestW <= B){
        for (int i = 0; i < N; i++) if (best[i]) chosen.push_back(i);
    } else {
        // fallback safety net (should not trigger given the planted guarantee):
        // seed the target's live cells, budget-capped.
        for (int i = 0; i < N && (int)chosen.size() < B; i++) if (target[i] == '1') chosen.push_back(i);
    }
    if ((int)chosen.size() > B) chosen.resize(B);

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++) printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    if (chosen.empty()) printf("\n");
    return 0;
}
