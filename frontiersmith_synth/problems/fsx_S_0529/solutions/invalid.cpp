// TIER: invalid
// Deliberately infeasible: references a node index far out of range -> checker rejects.
#include <bits/stdc++.h>
using namespace std;
int main(){
    printf("1\n0 1000000\n");
    return 0;
}
