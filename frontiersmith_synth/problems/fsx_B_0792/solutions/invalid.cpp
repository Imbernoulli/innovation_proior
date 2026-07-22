// TIER: invalid
// Deliberately infeasible: claims a barrier with a height far outside [1,HMAX].
// The checker's bounded ouf.readInt(1,100,"h") rejects this immediately (WA -> ratio 0).
#include <cstdio>
int main() {
    printf("1\n");
    printf("0 0 0 999999999\n");
    return 0;
}
