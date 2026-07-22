// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: the identity permutation is a valid bijection of
// {0,...,p-1} but leaves EVERY position unrewired (pi(x) = x for all x), which
// violates the derangement feasibility constraint. Must score 0.

int main(){
    int p;
    scanf("%d", &p);
    for (int x = 0; x < p; x++) printf("%d%c", x, x + 1 < p ? ' ' : '\n');
    return 0;
}
