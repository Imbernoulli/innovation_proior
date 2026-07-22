// TIER: invalid
// The classic TRAP: emit a full de Bruijn sequence of order K (an Eulerian circuit
// of the COMPLETE de Bruijn graph) that covers EVERY k-mer -- ignoring the
// forbidden set.  It therefore contains forbidden factors and is rejected
// (score 0).  This is exactly the recipe a solver "who knows de Bruijn sequences"
// writes first, and it is disqualified.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int A, K, F;
    scanf("%d %d %d", &A, &K, &F);
    int V = 1; for (int i = 0; i < K - 1; i++) V *= A;
    // (we intentionally ignore the forbidden list)
    static char buf[64];
    for (int t = 0; t < F; t++) scanf("%s", buf);

    // Hierholzer on the COMPLETE de Bruijn graph (every edge present).
    vector<int> ptr(V, 0);
    vector<int> stV, stS;
    stV.push_back(0); stS.push_back(-1);
    vector<int> revSyms;
    while (!stV.empty()) {
        int v = stV.back();
        if (ptr[v] < A) {
            int s = ptr[v]++;
            int nx = (int)(((ll)v * A + s) % V);
            stV.push_back(nx); stS.push_back(s);
        } else {
            int s = stS.back();
            stV.pop_back(); stS.pop_back();
            if (s != -1) revSyms.push_back(s);
        }
    }
    reverse(revSyms.begin(), revSyms.end());

    string S;
    S.reserve(revSyms.size() + K + 1);
    for (int i = 0; i < K - 1; i++) S.push_back('0');   // start vertex 0 = "aa..a"
    for (int s : revSyms) S.push_back((char)('0' + s));

    printf("%s\n", S.c_str());
    return 0;
}
