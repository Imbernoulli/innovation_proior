// TIER: invalid
// Deliberately infeasible: claim to protect a stand id far out of range. The checker's
// bounded read ouf.readInt(1,N,"vid") rejects it -> no Ratio -> scores 0.
#include <cstdio>
int main(){
    printf("1\n");
    printf("1 2000000000\n");
    return 0;
}
