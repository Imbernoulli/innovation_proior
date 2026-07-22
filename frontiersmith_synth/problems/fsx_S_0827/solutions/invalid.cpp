// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: one output row contains a character outside {0,1}.

int main(){
    int N, T; long long alpha, beta;
    if (!(cin >> N >> T >> alpha >> beta)) return 0;
    string rule; cin >> rule;
    for (int i = 0; i < N; i++){ string row; cin >> row; }

    for (int i = 0; i < N; i++){
        if (i == N / 2){
            string bad(N, '0');
            bad[0] = '9';
            cout << bad << "\n";
        } else {
            cout << string(N, '0') << "\n";
        }
    }
    return 0;
}
