// TIER: invalid
// Deliberately infeasible: places two parts on top of each other (overlap) ->
// checker must reject -> ratio 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    ll N, M, DA, DB, PW, PG, THR;
    if (!(cin >> N >> M >> DA >> DB >> PW >> PG >> THR)) return 0;
    vector<ll> w(M + 1), h(M + 1), v(M + 1), q(M + 1);
    for (int i = 1; i <= M; i++) cin >> w[i] >> h[i] >> v[i] >> q[i];
    if (M >= 2){
        // both at (0,0) -> guaranteed overlap
        printf("2\n");
        printf("1 0 0\n");
        printf("2 0 0\n");
    } else {
        printf("1\n1 0 0\n");   // if only one part, fall back (still feasible) -- rare
    }
    return 0;
}
