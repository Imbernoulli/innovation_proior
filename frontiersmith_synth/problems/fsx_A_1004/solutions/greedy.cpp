// TIER: greedy
// The "obvious" first attempt: grab the mmax highest-value interior slots
// (wherever they are) and hope the fold conditions work out. Uses a simple
// blocky mountain-block / valley-block labeling. If the naive pick is
// infeasible (very common once k is odd, or once the value table isn't
// already conveniently symmetric), falls back to the same safe midpoint
// construction as trivial.cpp -- it does NOT search for a feasible
// alternative that keeps any of the high-value slots.
#include <bits/stdc++.h>
using namespace std;

static bool kawasakiBalanced(int k, int S, const vector<int>& r) {
    int extra = (int)r.size();
    long long d = (long long)k * (extra + 1);
    vector<long long> pos;
    pos.reserve(d);
    for (int t = 0; t < k; t++) {
        pos.push_back((long long)t * S + 0);
        for (int x : r) pos.push_back((long long)t * S + x);
    }
    long long total = (long long)k * S;
    long long sumEven = 0, sumOdd = 0;
    for (long long i = 0; i < d; i++) {
        long long nxt = (i + 1 < d) ? pos[i + 1] : pos[0] + total;
        long long g = nxt - pos[i];
        if (i % 2 == 0) sumEven += g; else sumOdd += g;
    }
    return sumEven == sumOdd;
}

static void printBlocky(long long d) {
    for (long long i = 0; i < d; i++) {
        int lab = (i <= d / 2) ? 1 : 0; // first d/2+1 are M, rest V
        cout << (lab ? 'M' : 'V') << " \n"[i + 1 == d];
    }
}

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

static void printTrivialFallback(int k, int S, int mmax) {
    vector<int> rb = buildNaiveBaseline(k, S, mmax);
    long long d = (long long)k * ((long long)rb.size() + 1);
    cout << rb.size() << "\n";
    for (size_t i = 0; i < rb.size(); i++) cout << rb[i] << " \n"[i + 1 == rb.size()];
    if (rb.empty()) cout << "\n";
    for (long long i = 0; i < d; i++) {
        int lab = (i % 2 == 0) ? 1 : 0;
        if (i == d - 1) lab = 1;
        cout << (lab ? 'M' : 'V') << " \n"[i + 1 == d];
    }
}

int main() {
    int k, S, mmax, beta1000, gamma1000;
    cin >> k >> S >> mmax >> beta1000 >> gamma1000;
    vector<int> v(S, 0);
    for (int i = 1; i <= S - 1; i++) cin >> v[i];

    // rank slots by value, take top mmax (any positions), then sort ascending
    vector<int> order;
    for (int i = 1; i <= S - 1; i++) order.push_back(i);
    sort(order.begin(), order.end(), [&](int a, int b) { return v[a] > v[b]; });
    vector<int> r(order.begin(), order.begin() + mmax);
    sort(r.begin(), r.end());

    long long d = (long long)k * (mmax + 1);
    bool feasible = (d % 2 == 0) && kawasakiBalanced(k, S, r);

    if (!feasible) {
        printTrivialFallback(k, S, mmax);
        return 0;
    }
    cout << mmax << "\n";
    for (size_t i = 0; i < r.size(); i++) cout << r[i] << " \n"[i + 1 == r.size()];
    if (r.empty()) cout << "\n";
    printBlocky(d);
    return 0;
}
