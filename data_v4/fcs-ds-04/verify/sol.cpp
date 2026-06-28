#include <bits/stdc++.h>
using namespace std;

// Wavelet tree over the COMPRESSED value domain [0, sigma).
// Each internal node owns a contiguous value range [lo, hi]; it stores, for
// every position currently routed into this node, a prefix-count map[i] =
// number of the first i elements (in this node's order) that go LEFT, i.e.
// whose value id is <= mid. With that single prefix array per node we can, in
// O(1) per level, remap an index range [l, r) down to the correct child, which
// gives:
//   - rankLE(l, r, x): how many of positions [l, r) hold a value id <= x, and
//   - kth(l, r, k):    the k-th smallest (1-indexed) value id among [l, r).
// Both run in O(log sigma); the whole structure is O(n log sigma) space.
struct WaveletTree {
    int lo, hi;               // value-id range [lo, hi] this node covers
    WaveletTree *left = nullptr, *right = nullptr;
    vector<int> mp;           // mp[i] = #elements among first i (in this node) that go left

    WaveletTree(vector<int>::iterator from, vector<int>::iterator to, int x, int y) {
        lo = x; hi = y;
        if (lo == hi || from >= to) return;          // single value, or empty
        int mid = lo + (hi - lo) / 2;
        auto goLeft = [mid](int v) { return v <= mid; };
        mp.reserve((size_t)(to - from) + 1);
        mp.push_back(0);
        for (auto it = from; it != to; ++it)
            mp.push_back(mp.back() + (goLeft(*it) ? 1 : 0));
        // stable_partition keeps relative order within each child, which is what
        // the prefix map relies on when it routes an index range downward.
        auto pivot = stable_partition(from, to, goLeft);
        left  = new WaveletTree(from, pivot, lo, mid);
        right = new WaveletTree(pivot, to,   mid + 1, hi);
    }

    // Number of positions in [l, r) whose value id is <= x.  (0-indexed half-open.)
    int rankLE(int l, int r, int x) const {
        if (l >= r || x < lo) return 0;
        if (hi <= x) return r - l;                   // entire node qualifies
        int la = mp[l], ra = mp[r];                  // map endpoints into left child
        return left->rankLE(la, ra, x)
             + right->rankLE(l - la, r - ra, x);     // right index = original - #left
    }

    // k-th smallest (1-indexed) value id among positions [l, r).  Assumes 1<=k<=r-l.
    int kth(int l, int r, int k) const {
        if (lo == hi) return lo;                     // reached a single value
        int la = mp[l], ra = mp[r];
        int inLeft = ra - la;                        // how many of [l, r) go left
        if (k <= inLeft) return left->kth(la, ra, k);
        return right->kth(l - la, r - ra, k - inLeft);
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<long long> raw(n);
    for (auto &v : raw) cin >> v;

    // Coordinate-compress the values into ids [0, sigma).
    vector<long long> srt(raw.begin(), raw.end());
    sort(srt.begin(), srt.end());
    srt.erase(unique(srt.begin(), srt.end()), srt.end());
    int sigma = (int)srt.size();

    vector<int> comp(n);
    for (int i = 0; i < n; ++i)
        comp[i] = (int)(lower_bound(srt.begin(), srt.end(), raw[i]) - srt.begin());

    WaveletTree *root = nullptr;
    if (n > 0) root = new WaveletTree(comp.begin(), comp.end(), 0, sigma - 1);

    long long last = 0;                              // previous answer (XOR key)
    string out;
    out.reserve((size_t)q * 7);

    for (int t = 0; t < q; ++t) {
        long long type, A, B, C;
        cin >> type >> A >> B >> C;
        // Online: every parameter is XORed with the previous answer.
        A ^= last; B ^= last; C ^= last;

        long long ans;
        if (type == 1) {
            // count of positions in [l, r] (1-indexed, inclusive) with value <= x
            int l = (int)A, r = (int)B;
            long long x = C;
            // #compressed ids whose original value is <= x:
            int xid = (int)(upper_bound(srt.begin(), srt.end(), x) - srt.begin()) - 1;
            if (xid < 0) ans = 0;                    // x smaller than every value
            else ans = root->rankLE(l - 1, r, xid);
        } else {
            // k-th smallest in positions [l, r] (1-indexed inclusive)
            int l = (int)A, r = (int)B, k = (int)C;
            int id = root->kth(l - 1, r, k);
            ans = srt[id];                           // map id back to original value
        }
        out += to_string(ans);
        out += '\n';
        last = ans;                                  // answers are always >= 0 here
    }
    cout << out;
    return 0;
}
