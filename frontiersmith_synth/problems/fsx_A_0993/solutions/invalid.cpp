// TIER: invalid
// Deliberately infeasible: the very first diagonal-orientation token must
// be in {0,1}; printing 9 fails the checker's bounded read immediately ->
// no Ratio -> scores 0.
#include <cstdio>
int main(){
    printf("9 1 2\n");
    return 0;
}
