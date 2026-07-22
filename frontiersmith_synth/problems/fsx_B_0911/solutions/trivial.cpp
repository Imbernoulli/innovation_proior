// TIER: trivial
// Round-robin channel c_i = 1 + ((i-1) mod C), oblivious to interference and to
// ducting entirely. This is EXACTLY the checker's own internal baseline B, so it
// reproduces ratio ~0.1 on every case by construction.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, C, K;
    if (!(cin >> N >> C >> K)) return 0;
    for (int i = 1; i <= N; i++) {
        int ch = 1 + ((i - 1) % C);
        cout << ch << (i < N ? ' ' : '\n');
    }
    return 0;
}
