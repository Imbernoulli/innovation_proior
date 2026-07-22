// TIER: invalid
// Deliberately infeasible: the first token must be a node type in {0,1};
// printing 9 fails the checker's bounded read -> no Ratio -> scores 0.
#include <cstdio>
int main(){
    printf("9\n");
    return 0;
}
