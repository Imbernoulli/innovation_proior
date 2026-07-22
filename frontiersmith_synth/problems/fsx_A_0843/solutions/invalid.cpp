// TIER: invalid
// Deliberately infeasible: rotation must be in {0,1,2,3}; printing 9 fails the
// checker's bounded read on the very first block -> no Ratio -> scores 0.
#include <cstdio>
int main(){
    printf("0 0 9\n");
    return 0;
}
