// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Identity seeding: player i placed in slot i. This is exactly the checker's
// own internal baseline construction, so it always scores ratio ~= 0.1.
int main(){
    int N, K;
    if (!(cin >> N >> K)) return 0;
    for (int i = 0; i < N; i++) { string row; cin >> row; }
    for (int i = 0; i < K; i++) { int x; cin >> x; }
    for (int i = 0; i < K; i++) { int x; cin >> x; }
    for (int i = 1; i <= N; i++) cout << i << (i == N ? '\n' : ' ');
    return 0;
}
