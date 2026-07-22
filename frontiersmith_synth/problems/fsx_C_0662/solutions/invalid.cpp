// TIER: invalid
// Deliberately infeasible: claims to activate line 1 with an absurd participant
// count that can never fit the bounded read (2..k_g) -- no Ratio emitted, 0.
#include <cstdio>
int main(){
    printf("1\n1 999999999 1 2\n");
    return 0;
}
