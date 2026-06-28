#include <bits/stdc++.h>
using namespace std;

/*
    Parity-Invariant Reachability (generalized R x C sliding-tile puzzle).

    A board of R rows and C columns (R >= 2, C >= 2) holds the values
    0..R*C-1, where 0 is the blank. A move slides a tile orthogonally
    adjacent to the blank into the blank cell (the blank swaps with a
    4-neighbour). Given boards A and B (permutations of {0..R*C-1}), decide
    whether A can be turned into B. Print YES or NO.

    Insight (the whole problem): every move is ONE transposition of the
    cell-permutation (so it flips permutation parity) AND it moves the blank
    by Manhattan distance exactly 1 (so it flips the parity of the blank's
    Manhattan displacement). Hence the quantity

        parity(pi)  XOR  parity(Manhattan(blank_A, blank_B))

    where pi is the permutation carrying A's cell-contents to B's, is
    invariant under every move and equals 0 in the start state (A == A).
    For a genuine 2D board (R,C >= 2) this single invariant is also
    sufficient (classic 15-puzzle / Wilson's theorem). So:

        reachable  <=>  same multiset  AND  parity(pi) == parity(Manhattan).

    We compute parity(pi) in O(n) via cycle decomposition and read the blank
    positions directly. O(n) time, O(n) memory; n = R*C up to 1e6.
*/

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long R, C;
    if (!(cin >> R >> C)) return 0;
    long long n = R * C;

    vector<int> A(n), B(n);
    for (auto &x : A) cin >> x;
    for (auto &x : B) cin >> x;

    // Precondition: A and B must be permutations of the same multiset.
    // The contract guarantees both are permutations of {0..n-1}; we still
    // verify defensively so a malformed pair is reported NO, never crashes.
    auto valid_perm = [&](const vector<int> &V) -> bool {
        vector<char> seen(n, 0);
        for (int v : V) {
            if (v < 0 || v >= n || seen[v]) return false;
            seen[v] = 1;
        }
        return true;
    };
    if (!valid_perm(A) || !valid_perm(B)) {
        cout << "NO\n";
        return 0;
    }

    // posB[v] = cell index of value v in board B.
    vector<int> posB(n);
    for (int cell = 0; cell < n; cell++) posB[B[cell]] = cell;

    // pi[cell] = the cell that A[cell]'s value occupies in B.
    // parity(pi) = (n - number_of_cycles) mod 2.
    vector<int> pi(n);
    for (int cell = 0; cell < n; cell++) pi[cell] = posB[A[cell]];

    vector<char> visited(n, 0);
    long long cycles = 0;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            cycles++;
            int j = i;
            while (!visited[j]) {
                visited[j] = 1;
                j = pi[j];
            }
        }
    }
    int parPi = (int)((n - cycles) & 1LL);

    // Manhattan distance between the two blank cells, parity only.
    int za = 0, zb = 0;
    for (int cell = 0; cell < n; cell++) {
        if (A[cell] == 0) za = cell;
        if (B[cell] == 0) zb = cell;
    }
    long long ra = za / C, ca = za % C;
    long long rb = zb / C, cb = zb % C;
    int parMan = (int)((llabs(ra - rb) + llabs(ca - cb)) & 1LL);

    cout << (parPi == parMan ? "YES" : "NO") << "\n";
    return 0;
}
