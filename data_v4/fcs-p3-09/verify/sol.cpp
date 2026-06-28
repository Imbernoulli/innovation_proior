#include <bits/stdc++.h>
using namespace std;

// Number of domino tilings of a 3 x N board, modulo m.
// We use the column-profile transfer matrix on the 8 = 2^3 protrusion states.
// answer = (T^N)[0][0] mod m, where state bit r means a horizontal domino sticks
// out of row r into the next column. Starting and ending state is 0 (no protrusion).

static long long MOD;

struct Mat {
    static const int S = 8;
    unsigned long long a[8][8];
    Mat() { for (int i = 0; i < S; i++) for (int j = 0; j < S; j++) a[i][j] = 0; }
};

Mat mul(const Mat &A, const Mat &B) {
    Mat C;
    for (int i = 0; i < Mat::S; i++) {
        for (int k = 0; k < Mat::S; k++) {
            unsigned long long aik = A.a[i][k];
            if (!aik) continue;
            for (int j = 0; j < Mat::S; j++) {
                if (B.a[k][j])
                    C.a[i][j] = (C.a[i][j] + aik * B.a[k][j]) % (unsigned long long)MOD;
            }
        }
    }
    return C;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long N, m;
    if (!(cin >> N >> m)) return 0;
    MOD = m;

    // Reduce everything mod m up front; m can be 1, giving answer 0.
    // Odd N never tiles (each domino covers an even area; 3*N odd => impossible).
    if (N % 2 == 1) {
        cout << 0 % m << "\n";
        return 0;
    }

    // Build the 8x8 transfer matrix T[next][cur] = number of ways to completely fill
    // one column whose prefilled cells are 'cur', leaving protrusion 'next' on the right.
    Mat T;
    for (int cur = 0; cur < 8; cur++) {
        // enumerate fillings of this column row by row
        // recursion implemented with an explicit stack to keep it self-contained
        // state: (row, filled, nxt)
        vector<array<int,3>> st;
        st.push_back({0, cur, 0});
        while (!st.empty()) {
            auto [r, filled, nxt] = st.back();
            st.pop_back();
            if (r == 3) { T.a[nxt][cur] = (T.a[nxt][cur] + 1) % (unsigned long long)MOD; continue; }
            if (filled & (1 << r)) { st.push_back({r + 1, filled, nxt}); continue; }
            // vertical domino covering rows r and r+1 (both must be free)
            if (r + 1 < 3 && !(filled & (1 << (r + 1))))
                st.push_back({r + 2, filled | (1 << r) | (1 << (r + 1)), nxt});
            // horizontal domino sticking out of row r into the next column
            st.push_back({r + 1, filled | (1 << r), nxt | (1 << r)});
        }
    }

    // R = T^N via fast exponentiation; answer = R[0][0].
    Mat R;
    for (int i = 0; i < 8; i++) R.a[i][i] = 1 % (unsigned long long)MOD;
    long long e = N;
    Mat base = T;
    while (e > 0) {
        if (e & 1) R = mul(R, base);
        base = mul(base, base);
        e >>= 1;
    }

    cout << (R.a[0][0] % (unsigned long long)MOD) << "\n";
    return 0;
}
