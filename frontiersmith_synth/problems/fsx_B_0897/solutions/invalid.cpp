// TIER: invalid
// Deliberately infeasible: claims zero bridges, so every island is disconnected
// from the frame. The checker's connectivity check must reject this -> score 0.
#include <cstdio>
int main(){
    printf("0\n");
    return 0;
}
