// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
// Deliberately infeasible: couples every P-node to the SAME C-node (id 1), which
// is not a permutation. The checker must reject this with score 0.
int main() {
    int N;
    if (!(cin >> N)) return 0;
    for (int i = 1; i <= N; i++) printf("1\n");
    return 0;
}
