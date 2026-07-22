// TIER: invalid
// Deliberately infeasible: repeats an index (duplicate team member) and also
// includes an out-of-range index. The checker must score this 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    printf("3\n0 0 5000000\n");
    return 0;
}
