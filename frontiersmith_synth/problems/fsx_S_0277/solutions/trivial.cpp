// TIER: trivial
// Do-nothing baseline: shut no junction. F = B, so ratio = 0.1 exactly.
#include <bits/stdc++.h>
using namespace std;
int main() {
    // We do not even need to parse the graph: an empty shutdown set is always feasible.
    printf("0\n");
    return 0;
}
