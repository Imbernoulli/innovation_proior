// TIER: invalid
// Deliberately infeasible: assigns the same drone twice -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    // ignore input entirely; emit an infeasible assignment
    printf("2\n1 1\n1 1\n");
    return 0;
}
