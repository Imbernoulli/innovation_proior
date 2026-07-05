#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, L;
vector<int> eu, ev;
vector<ll> ew;

// cut weight of a 0/1 side labelling (side[i] in {0,1})
ll cutWeight(const vector<int>& side) {
    ll c = 0;
    for (int i = 0; i < m; i++)
        if (side[eu[i]] != side[ev[i]]) c += ew[i];
    return c;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    L = inf.readInt();

    eu.resize(m); ev.resize(m); ew.resize(m);
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ew[i] = inf.readInt();
    }

    // ---- internal baseline: contiguous split {1..floor(n/2)} vs rest ----
    int h = n / 2;
    vector<int> base(n + 1, 0);
    for (int i = 1; i <= n; i++) base[i] = (i <= h) ? 0 : 1;
    ll B = cutWeight(base);
    if (B <= 0) quitf(_fail, "bad instance: baseline cut B=%lld", B);

    // ---- read & validate participant's Habitat-A set ----
    int a = ouf.readInt(0, n, "a");
    vector<char> inA(n + 1, 0);
    for (int i = 0; i < a; i++) {
        int id = ouf.readInt(1, n, "crewId");
        if (inA[id]) quitf(_wa, "crew %d listed in Habitat A more than once", id);
        inA[id] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // headcount balance
    int b = n - a;
    if (llabs((ll)a - (ll)b) > L)
        quitf(_wa, "unbalanced split: |%d - %d| = %d > L = %d", a, b, abs(a - b), L);

    vector<int> side(n + 1, 0);
    for (int i = 1; i <= n; i++) side[i] = inA[i] ? 0 : 1;
    ll F = cutWeight(side);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
