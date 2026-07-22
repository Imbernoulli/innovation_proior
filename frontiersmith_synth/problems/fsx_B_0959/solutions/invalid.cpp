// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: prints N copies of 0, which is out of the required
// range [1, N] for a slot assignment -- the checker's bounded readInt must
// reject this immediately, scoring 0.
int main(){
    int N, K;
    if (!(cin >> N >> K)) return 0;
    for (int i = 0; i < N; i++) { string row; cin >> row; }
    for (int i = 0; i < K; i++) { int x; cin >> x; }
    for (int i = 0; i < K; i++) { int x; cin >> x; }
    for (int i = 0; i < N; i++) cout << 0 << (i + 1 == N ? '\n' : ' ');
    return 0;
}
