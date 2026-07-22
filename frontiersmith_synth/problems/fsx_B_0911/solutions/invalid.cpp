// TIER: invalid
// Deliberately infeasible: emits channel 0 for every station (out of [1,C]) ->
// checker's ouf.readInt(1,C,...) rejects it -> scores 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, C, K;
    if (!(cin >> N >> C >> K)) return 0;
    for (int i = 1; i <= N; i++) cout << 0 << (i < N ? ' ' : '\n');
    return 0;
}
