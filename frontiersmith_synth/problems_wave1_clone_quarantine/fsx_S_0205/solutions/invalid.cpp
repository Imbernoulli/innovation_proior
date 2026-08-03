// TIER: invalid
// Deliberately infeasible: claims to shut a valve whose index is out of range, so the
// checker's bounded read rejects it and the output scores 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    printf("1\n1000000000\n");
    return 0;
}
