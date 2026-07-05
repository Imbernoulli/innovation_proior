// TIER: invalid
// Emits a move with an out-of-range destination position -> infeasible -> 0.
#include <cstdio>
int main(){
    printf("1\n1 999999\n");
    return 0;
}
