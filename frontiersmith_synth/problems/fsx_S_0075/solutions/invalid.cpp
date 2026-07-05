// TIER: invalid
// Deliberately infeasible: references a pod design index that does not exist.
#include <bits/stdc++.h>
using namespace std;
int main(){
    // Ignore input; emit one pod claiming design 1000000 -> out of range -> scores 0.
    printf("1\n1000000 0 0 1 0 0 1\n");
    return 0;
}
