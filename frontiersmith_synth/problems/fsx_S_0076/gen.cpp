#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static inline long long man(long long ax,long long ay,long long bx,long long by){
    return llabs(ax-bx)+llabs(ay-by);
}

int main(int argc,char*argv[]){
    registerGen(argc,argv,1);
    int t=atoi(argv[1]);

    // number of tasks: tiny at t=1, growing to ~58 at t=10 (small scale)
    int m = (t==1)?4 : (4 + 6*(t-1));
    if(m>60) m=60;

    int C=1000;
    int Xd = rnd.next(0,C), Yd = rnd.next(0,C);

    // structural mode -> variety across the ladder
    int mode = (t-1)%3; // 0 uniform, 1 separated pickup/delivery regions, 2 outliers

    printf("%d %d\n", Xd, Yd);
    printf("%d\n", m);

    for(int j=0;j<m;j++){
        long long px,py,dx,dy;
        if(mode==0){
            px=rnd.next(0,C); py=rnd.next(0,C);
            dx=rnd.next(0,C); dy=rnd.next(0,C);
        } else if(mode==1){
            // pickups clustered low, deliveries clustered high -> precedence tension
            px=rnd.next(0,C/2); py=rnd.next(0,C/2);
            dx=rnd.next(C/2,C); dy=rnd.next(C/2,C);
        } else {
            // mostly local pairs but ~1/4 far-flung outliers (skip candidates)
            if(rnd.next(0,3)==0){
                px=rnd.next(0,C); py=rnd.next(0,C);
                dx=rnd.next(0,C); dy=rnd.next(0,C);
            } else {
                px=rnd.next(0,C); py=rnd.next(0,C);
                dx=px+rnd.next(-120,120); dy=py+rnd.next(-120,120);
                dx=max(0LL,min((long long)C,dx)); dy=max(0LL,min((long long)C,dy));
            }
        }
        // standalone round-trip cost of doing this task alone
        long long base = man(Xd,Yd,px,py)+man(px,py,dx,dy)+man(dx,dy,Xd,Yd);
        if(base<1) base=1;
        long long lo=(long long)llround(0.4*base);
        long long hi=(long long)llround(1.6*base);
        if(lo<1) lo=1;
        if(hi<lo) hi=lo+1;
        long long w=rnd.next(lo,hi);
        printf("%lld %lld %lld %lld %lld\n", px,py,dx,dy,w);
    }
    return 0;
}
