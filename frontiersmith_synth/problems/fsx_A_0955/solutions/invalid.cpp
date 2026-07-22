// TIER: invalid
// Deliberately infeasible: the first path length must be in [2, M]; printing 0
// fails the checker's bounded read immediately -> no Ratio -> scores 0.
#include <cstdio>
int main(){
    printf("0\n");
    return 0;
}
