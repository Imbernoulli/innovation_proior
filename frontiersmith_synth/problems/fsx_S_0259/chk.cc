#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, D;
vector<int> eu, ev;
vector<ll> ew;

int par[2000005], rnk[2000005];
int find(int x){ while(par[x]!=x){ par[x]=par[par[x]]; x=par[x];} return x; }
void uni(int a,int b){ a=find(a); b=find(b); if(a==b) return; if(rnk[a]<rnk[b]) swap(a,b); par[b]=a; if(rnk[a]==rnk[b]) rnk[a]++; }

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    D = inf.readInt();

    eu.resize(m + 1); ev.resize(m + 1); ew.resize(m + 1);
    // min-weight backbone edge for each consecutive pair (i,i+1)
    vector<ll> bbw(n + 1, -1);
    for (int i = 1; i <= m; i++) {
        int u = inf.readInt();
        int v = inf.readInt();
        ll w = inf.readInt();
        eu[i] = u; ev[i] = v; ew[i] = w;
        if (abs(u - v) == 1) {
            int lo = min(u, v);
            if (bbw[lo] < 0 || w < bbw[lo]) bbw[lo] = w;
        }
    }

    // internal baseline B = cost of the guaranteed feasible backbone path
    ll B = 0;
    for (int i = 1; i < n; i++) {
        if (bbw[i] < 0) quitf(_fail, "bad instance: missing backbone edge (%d,%d)", i, i + 1);
        B += bbw[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's network (edge index set) ----
    int r = ouf.readInt(0, m, "r");
    vector<char> used(m + 1, 0);
    vector<int> deg(n + 1, 0);
    for (int i = 1; i <= n; i++){ par[i] = i; rnk[i] = 0; }
    ll F = 0;
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "edgeIndex");
        if (used[idx]) quitf(_wa, "pipe index %d used more than once", idx);
        used[idx] = 1;
        int u = eu[idx], v = ev[idx];
        deg[u]++; deg[v]++;
        if (deg[u] > D) quitf(_wa, "junction %d exceeds fan-out limit %d", u, D);
        if (deg[v] > D) quitf(_wa, "junction %d exceeds fan-out limit %d", v, D);
        uni(u, v);
        F += ew[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // connectivity: all n junctions in one component
    int root = find(1);
    for (int i = 2; i <= n; i++)
        if (find(i) != root) quitf(_wa, "network is not connected: junction %d unreachable from junction 1", i);

    if (F <= 0) quitf(_wa, "empty / zero-cost network");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
