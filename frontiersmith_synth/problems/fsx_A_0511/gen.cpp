#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ============================================================================
// Family: debruijn-forbidden-cover  (theme: cover k-mers, dodge forbidden factors)
//
// de Bruijn graph B(a,k): vertices = (k-1)-mers (V=a^(k-1)); edges = k-mers
// (E=a^k), edge c=v*a+s goes v -> (v*a+s) mod V.  A FORBIDDEN FACTOR is a deleted
// edge.  Here edges are deleted INDIVIDUALLY at random (plus extra starving of the
// low-index region), so the surviving "allowed" graph is deliberately UN-BALANCED:
// many vertices become local sinks.
//
// Why this is a trap:
//   * A single greedy walk (append the smallest legal symbol) stalls the moment it
//     reaches a vertex whose allowed out-edges are all used -- i.e. at a sink -- and
//     it cannot return to pick up edges "behind" it.  It therefore strands a large
//     fraction of the reachable k-mers.
//   * The insight is to view coverage as a MAX-EDGE walk on the mutilated graph:
//     take the largest strongly-connected component, Eulerize it by repeating a few
//     allowed edges (which cost nothing, since coverage counts DISTINCT factors),
//     and sweep every one of its edges with a single Eulerian circuit.
//
// testId is a difficulty ladder (alphabet a, order k grow; deletion pattern varies).
// ============================================================================

static int A, K, V, E;
static vector<char> del;
static inline int tgt(int v, int s) { return (v * A + s) % V; }

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int aTab[11] = {0, 2, 2, 3, 3, 4, 5, 4, 5, 6, 4};
    int kTab[11] = {0, 4, 6, 4, 5, 5, 4, 6, 5, 5, 7};
    A = aTab[testId];
    K = kTab[testId];
    V = 1; for (int i = 0; i < K - 1; i++) V *= A;
    E = V * A;

    del.assign(E, 0);

    // ---- random individual edge deletion (unbalances the graph) ----
    double p = 0.16 + 0.01 * ((testId * 3) % 6);   // ~0.16 .. 0.21
    const char* env = getenv("PDEL");
    if (env) p = atof(env);
    for (int c = 0; c < E; c++)
        if (rnd.next(0.0, 1.0) < p) del[c] = 1;

    // ---- guarantee at least one forbidden factor and at least one allowed edge ----
    int deleted = 0; for (int c = 0; c < E; c++) deleted += del[c];
    if (deleted == 0) { del[0] = 1; deleted = 1; }
    if (deleted == E) { del[0] = 0; deleted = E - 1; }

    // ---- emit forbidden list, shuffled ----
    vector<int> forb; forb.reserve(deleted);
    for (int c = 0; c < E; c++) if (del[c]) forb.push_back(c);
    shuffle(forb.begin(), forb.end());

    vector<ll> pw(K + 1, 1);
    for (int i = 1; i <= K; i++) pw[i] = pw[i - 1] * A;

    string out;
    out.reserve((size_t)deleted * (K + 1) + 32);
    char line[64];
    int hlen = sprintf(line, "%d %d %d\n", A, K, (int)forb.size());
    out.append(line, hlen);
    for (int c : forb) {
        for (int i = 0; i < K; i++) {
            int dig = (int)((c / pw[K - 1 - i]) % A);
            out.push_back((char)('0' + dig));
        }
        out.push_back('\n');
    }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
