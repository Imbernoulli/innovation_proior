// TIER: trivial
// The checker's own naive (no value information) baseline construction: a
// fill of ~mmax/2 interior creases picked purely by index order (never by
// value), sized/parity-adjusted per k so it is always Kawasaki-feasible
// (even k: plain even-count index fill; odd k: midpoint + naive low-index
// mirror pairs). This reproduces the checker's internal baseline B exactly.
#include <bits/stdc++.h>
using namespace std;

static vector<int> buildNaiveBaseline(int k, int S, int mmax) {
    int target = mmax / 2;
    int mid = S / 2;
    vector<int> rb;
    if (k % 2 == 0) {
        int extra_b = target - (target % 2);
        for (int i = 1; i <= extra_b; i++) rb.push_back(i);
    } else {
        int extra_b = (target % 2 == 0) ? max(1, target - 1) : target;
        int numpairs = (extra_b - 1) / 2; // always <= mid-1 given k>=2,S>=6,mmax<=S-1
        rb.push_back(mid);
        for (int j = 1; j <= numpairs; j++) { rb.push_back(j); rb.push_back(S - j); }
        sort(rb.begin(), rb.end());
    }
    if (rb.empty()) rb.push_back(mid);
    return rb;
}

int main() {
    int k, S, mmax, beta1000, gamma1000;
    cin >> k >> S >> mmax >> beta1000 >> gamma1000;
    vector<int> v(S, 0);
    for (int i = 1; i <= S - 1; i++) cin >> v[i];

    vector<int> rb = buildNaiveBaseline(k, S, mmax);
    long long d = (long long)k * ((long long)rb.size() + 1);

    cout << rb.size() << "\n";
    for (size_t i = 0; i < rb.size(); i++) cout << rb[i] << " \n"[i + 1 == rb.size()];
    if (rb.empty()) cout << "\n";
    for (long long i = 0; i < d; i++) {
        int lab = (i % 2 == 0) ? 1 : 0; // M,V,M,V,...
        if (i == d - 1) lab = 1;        // flip last V -> M for |M-V|=2
        cout << (lab ? 'M' : 'V') << " \n"[i + 1 == d];
    }
    return 0;
}
