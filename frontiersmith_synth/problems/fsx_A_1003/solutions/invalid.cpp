// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: telescope 0 claims a visit to target index
// 1000000000, which is out of range [0, M-1] for any generated instance
// (M <= 320). The checker's bounded read must reject this -> score 0.

int main() {
    cout << 1 << "\n" << 0 << " " << 1000000000 << "\n";
    return 0;
}
