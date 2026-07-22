// TIER: invalid
// Deliberately infeasible: install one reflector with an out-of-range index.
// The checker's bounded read ouf.readInt(1, N, ...) rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    // count = 1 (<= K, since K >= 2), then an index far outside [1, N].
    printf("1\n1000000000\n");
    return 0;
}
