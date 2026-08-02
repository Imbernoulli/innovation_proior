// TIER: invalid
// Deliberately infeasible: claims a MOVE from a node index that is always out of range.
// The checker's bounded read (or the token/edge legality check) must reject this -> score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    cout << "MOVE 999999999 999999999\n";
    return 0;
}
