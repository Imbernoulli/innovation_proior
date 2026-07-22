// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: assigns job 1 to two different batches. The checker
// must reject this (job covered more than once) regardless of n, V or any
// other instance parameter, so this ladder rung is robust across all tests.
int main() {
    int n; long long V, LAMBDA;
    cin >> n >> V >> LAMBDA;
    // (ignore the rest of the input; the output is invalid no matter what)
    printf("1\n2 1 1\n");
    return 0;
}
