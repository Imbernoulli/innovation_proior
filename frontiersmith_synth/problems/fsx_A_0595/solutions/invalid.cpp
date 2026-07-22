// TIER: invalid
// Claims a demand index that does not exist (out of [0,D]) -> checker rejects -> 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    // whole sheet as a leaf claiming a nonexistent demand -> infeasible
    printf("0 999999\n");
    return 0;
}
