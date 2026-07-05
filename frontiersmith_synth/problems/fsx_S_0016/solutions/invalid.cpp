// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
int main() {
    // Deliberately infeasible: deliver request 1 with no prior pickup (precedence violation).
    // P >= 2 always, so index 1 exists. Must score 0.
    printf("1\n");
    printf("1 1\n");
    return 0;
}
