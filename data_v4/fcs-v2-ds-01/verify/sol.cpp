#include <bits/stdc++.h>
using namespace std;

// Map (x, y) on a side x side grid (side = 2^order) to its 1-D distance along the
// Hilbert space-filling curve. Canonical iterative xy2d construction.
static inline uint64_t hilbertOrder(uint32_t x, uint32_t y, int order) {
    uint64_t d = 0;
    for (int s = order - 1; s >= 0; --s) {
        uint32_t rx = (x >> s) & 1u;
        uint32_t ry = (y >> s) & 1u;
        d += ((uint64_t)((3u * rx) ^ ry)) << (2 * s);
        // rotate / reflect the quadrant
        if (ry == 0) {
            if (rx == 1) {
                uint32_t mask = (1u << order) - 1u;   // side - 1
                x = (mask - x) & mask;
                y = (mask - y) & mask;
            }
            uint32_t t = x; x = y; y = t;             // swap
        }
    }
    return d;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> a(n);
    int maxv = 0;
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
        if (a[i] > maxv) maxv = a[i];
    }

    // Hilbert grid order: smallest power of two side covering all indices.
    int order = 1;
    while ((1 << order) < max(1, n)) ++order;

    struct Query { int l, r, idx; uint64_t h; };
    vector<Query> qs(q);
    for (int i = 0; i < q; ++i) {
        int l, r;
        cin >> l >> r;          // 1-based inclusive
        --l; --r;               // to 0-based inclusive
        qs[i].l = l;
        qs[i].r = r;
        qs[i].idx = i;
        qs[i].h = hilbertOrder((uint32_t)l, (uint32_t)r, order);
    }

    sort(qs.begin(), qs.end(),
         [](const Query& A, const Query& B) { return A.h < B.h; });

    vector<int> cnt(maxv + 2, 0);
    vector<long long> ans(q, 0);

    long long cur = 0;          // sum over values v of cnt[v]^2 inside [curL, curR]
    int curL = 0, curR = -1;    // empty window

    auto add = [&](int pos) {
        int v = a[pos];
        long long c = cnt[v];
        cur += 2 * c + 1;       // (c+1)^2 - c^2
        cnt[v] = (int)(c + 1);
    };
    auto remove = [&](int pos) {
        int v = a[pos];
        long long c = cnt[v];
        cur -= 2 * c - 1;       // c^2 - (c-1)^2
        cnt[v] = (int)(c - 1);
    };

    for (const auto& Q : qs) {
        int L = Q.l, R = Q.r;
        while (curR < R) add(++curR);
        while (curL > L) add(--curL);
        while (curR > R) remove(curR--);
        while (curL < L) remove(curL++);
        ans[Q.idx] = cur;
    }

    string out;
    out.reserve((size_t)q * 12);
    char buf[24];
    for (int i = 0; i < q; ++i) {
        int len = sprintf(buf, "%lld\n", ans[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
