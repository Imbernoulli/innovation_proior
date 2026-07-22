// TIER: trivial
// Cuts nothing, glues nothing. Reproduces the checker's own do-nothing baseline
// exactly, so this scores ratio == 0.100000 on every test.
#include <bits/stdc++.h>
using namespace std;

int main(){
    // Consume the whole input so the pipe never breaks on any host, then emit M=0.
    int Ws, Hs; cin >> Ws >> Hs;
    for (int i = 0; i < Ws*Hs; i++){ int v; cin >> v; }
    int Wp, Hp; cin >> Wp >> Hp;
    for (int y = 0; y < Hp; y++){ string row; cin >> row; }
    for (int i = 0; i < Wp*Hp; i++){ int v; cin >> v; }
    int K; cin >> K;
    for (int i = 0; i < K; i++){ int nc; cin >> nc; for (int j = 0; j < nc; j++){ int a,b; cin >> a >> b; } }
    long long lambda, mu, Cu; cin >> lambda >> mu >> Cu;

    cout << 0 << "\n";
    return 0;
}
