// TIER: invalid
// Deliberately infeasible: shape_id is far outside [0, K-1], so the checker's
// own bounded read rejects it immediately. Must score 0 on every test.
#include <bits/stdc++.h>
using namespace std;
int main(){
    cout << 1 << "\n";
    cout << 999999 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << "\n";
    return 0;
}
