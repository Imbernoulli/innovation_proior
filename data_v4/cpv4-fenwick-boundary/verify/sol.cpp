#include <bits/stdc++.h>
using namespace std;

int N;                 // size of the Fenwick tree (number of distinct badge values)
vector<int> fen;       // 1-indexed Fenwick storing present-counts over compressed badges

void fadd(int i, int delta) {       // i is 1-indexed
    for (; i <= N; i += i & (-i)) fen[i] += delta;
}
long long fsum(int i) {             // sum over compressed positions [1..i], i may be 0
    long long s = 0;
    for (; i > 0; i -= i & (-i)) s += fen[i];
    return s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    struct Op { char type; long long x, y; };
    vector<Op> ops(q);
    vector<long long> vals;                 // badge values that actually get inserted/removed
    for (int i = 0; i < q; i++) {
        string t;
        cin >> t;
        if (t == "+" || t == "-") {
            long long b; cin >> b;
            ops[i] = {t[0], b, 0};
            vals.push_back(b);
        } else {                            // "?"
            long long lo, hi; cin >> lo >> hi;
            ops[i] = {'?', lo, hi};
        }
    }

    // Coordinate-compress only the badge values that appear in + / - events.
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    N = (int)vals.size();
    fen.assign(N + 1, 0);

    // 1-indexed compressed position of an existing badge value b (b must be in vals).
    auto posOf = [&](long long b) -> int {
        int idx = (int)(lower_bound(vals.begin(), vals.end(), b) - vals.begin());
        return idx + 1;                     // shift to 1-indexed
    };
    // Number of compressed values that are <= v (v is an arbitrary query bound).
    auto cntLE = [&](long long v) -> int {
        return (int)(upper_bound(vals.begin(), vals.end(), v) - vals.begin());
    };
    // Number of compressed values that are < v.
    auto cntLT = [&](long long v) -> int {
        return (int)(lower_bound(vals.begin(), vals.end(), v) - vals.begin());
    };

    string out;
    for (auto &op : ops) {
        if (op.type == '+') {
            fadd(posOf(op.x), +1);
        } else if (op.type == '-') {
            fadd(posOf(op.x), -1);
        } else {
            long long lo = op.x, hi = op.y;
            if (lo > hi) { out += "0\n"; continue; }
            // present badges with lo <= badge <= hi (inclusive both ends).
            // In compressed space: positions whose value is <= hi, minus those whose value is < lo.
            int rHi = cntLE(hi);            // compressed index of last value <= hi
            int rLoExclusive = cntLT(lo);   // count of values strictly < lo
            long long ans = fsum(rHi) - fsum(rLoExclusive);
            out += to_string(ans);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
