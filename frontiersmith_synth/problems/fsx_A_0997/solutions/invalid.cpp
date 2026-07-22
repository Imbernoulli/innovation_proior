// TIER: invalid
// Deliberately infeasible: claims to select a member index that is far outside
// the valid candidate range [0, M-1]. The checker must reject this (score 0).
#include <bits/stdc++.h>
using namespace std;
int main() {
    printf("1\n");
    printf("999999 0\n");
    return 0;
}
