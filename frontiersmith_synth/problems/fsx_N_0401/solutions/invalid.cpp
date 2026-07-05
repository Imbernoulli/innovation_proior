// TIER: invalid
// Deliberately infeasible: cut an out-of-range coupler index -> must score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    printf("1\n");
    printf("2000000000\n");   // index > m, fails bounded read in checker
    return 0;
}
