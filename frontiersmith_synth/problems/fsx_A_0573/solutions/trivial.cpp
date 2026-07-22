// TIER: trivial
// Install nothing: every listener hears only the direct pulse g0, so
// F = M*g0 = B and ratio = 0.1 exactly (the checker baseline).
#include <bits/stdc++.h>
using namespace std;
int main(){
    // We do not even need to parse the field; the empty install is always feasible.
    printf("0\n");
    return 0;
}
