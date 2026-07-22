// TIER: invalid
// Deliberately infeasible: dice the finest grid (every cell 1x1) but claim a
// die whose dimensions are strictly larger than 1x1 on EVERY cell. No die
// other than the 1x1 base unit can ever fit a 1x1 rectangle, so the checker
// must reject the very first assignment -> score 0 on every test.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, K;
    cin >> N >> K;
    vector<int> A(K + 1), B(K + 1);
    vector<ll> V(K + 1);
    for (int i = 1; i <= K; i++) cin >> A[i] >> B[i] >> V[i];
    int D;
    cin >> D;
    for (int i = 0; i < D; i++) { int r, c; cin >> r >> c; }

    int bad = 1;
    for (int i = 1; i <= K; i++)
        if (!(A[i] == 1 && B[i] == 1)) { bad = i; break; }

    cout << (N - 1);
    for (int p = 1; p <= N - 1; p++) cout << ' ' << p;
    cout << '\n';
    cout << (N - 1);
    for (int p = 1; p <= N - 1; p++) cout << ' ' << p;
    cout << '\n';

    for (int r = 1; r <= N; r++)
        for (int c = 1; c <= N; c++)
            cout << bad << ' ';
    cout << '\n';
    return 0;
}
