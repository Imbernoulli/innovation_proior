#include <bits/stdc++.h>
using namespace std;

static const int BITS = 60;

struct XorBasis {
    // basis[b] holds a vector whose highest set bit is exactly b (or 0 if empty).
    unsigned long long basis[BITS];
    int rank;                 // number of independent vectors inserted
    bool reduced;             // is the basis currently in reduced row-echelon form?
    XorBasis() { memset(basis, 0, sizeof(basis)); rank = 0; reduced = true; }

    // Insert x; returns true iff x was independent (rank increased).
    bool insert(unsigned long long x) {
        for (int b = BITS - 1; b >= 0; --b) {
            if (!((x >> b) & 1ULL)) continue;
            if (!basis[b]) {
                basis[b] = x;
                ++rank;
                reduced = false;   // a fresh pivot may need cleaning before order queries
                return true;
            }
            x ^= basis[b];
        }
        return false;              // x reduced to 0: dependent
    }

    // Reduce to row-echelon form: every pivot bit appears in exactly one basis vector.
    void makeReduced() {
        if (reduced) return;
        for (int b = 0; b < BITS; ++b) {
            if (!basis[b]) continue;
            for (int c = b + 1; c < BITS; ++c) {
                if (basis[c] && ((basis[c] >> b) & 1ULL))
                    basis[c] ^= basis[b];
            }
        }
        reduced = true;
    }

    // Maximum XOR over the span (empty subset -> 0 included automatically).
    unsigned long long maxXor() const {
        unsigned long long r = 0;
        for (int b = BITS - 1; b >= 0; --b)
            if (basis[b] && (r ^ basis[b]) > r) r ^= basis[b];
        return r;
    }

    // Is x in the span (representable as a subset-XOR)?
    bool representable(unsigned long long x) const {
        for (int b = BITS - 1; b >= 0; --b) {
            if (!((x >> b) & 1ULL)) continue;
            if (!basis[b]) return false;
            x ^= basis[b];
        }
        return x == 0;
    }

    // k-th smallest distinct value (0-indexed: k=0 -> smallest = 0).
    // Requires reduced form; valid range 0 <= k < 2^rank.
    unsigned long long kthSmallest(unsigned long long k) {
        makeReduced();
        unsigned long long res = 0;
        int idx = 0;
        for (int b = 0; b < BITS; ++b) {
            if (!basis[b]) continue;
            if ((k >> idx) & 1ULL) res ^= basis[b];
            ++idx;
        }
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    XorBasis B;
    string out;
    out.reserve(1 << 16);

    for (int i = 0; i < q; ++i) {
        int type;
        cin >> type;
        if (type == 1) {
            unsigned long long x;
            cin >> x;
            B.insert(x);
            // no output for an add
        } else if (type == 2) {
            // maximum subset XOR
            out += to_string(B.maxXor());
            out += '\n';
        } else if (type == 3) {
            unsigned long long x;
            cin >> x;
            out += (B.representable(x) ? "YES\n" : "NO\n");
        } else { // type == 4: k-th smallest distinct subset-XOR value, 1-indexed
            unsigned long long k;
            cin >> k;
            // #distinct values = 2^rank; rank <= 60 since values are <= 60 bits.
            unsigned long long total = (1ULL << B.rank);
            // k is 1-indexed; valid iff 1 <= k <= 2^rank, else report -1.
            if (k >= 1 && k <= total) {
                out += to_string(B.kthSmallest(k - 1));
                out += '\n';
            } else {
                out += "-1\n";
            }
        }
    }

    cout << out;
    return 0;
}
