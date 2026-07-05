#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static long long manh(long long x1,long long y1,long long x2,long long y2){
    return llabs(x1-x2)+llabs(y1-y2);
}

int main(int argc,char*argv[]){
    registerTestlibCmd(argc,argv);

    int P = inf.readInt();
    long long Q = inf.readLong();
    long long K = inf.readLong();
    long long x0 = inf.readLong(), y0 = inf.readLong();

    vector<long long> ax(P+1),ay(P+1),bx(P+1),by(P+1),m(P+1),c(P+1),w(P+1);
    long long B = 0;
    for(int i=1;i<=P;i++){
        ax[i]=inf.readLong(); ay[i]=inf.readLong();
        bx[i]=inf.readLong(); by[i]=inf.readLong();
        m[i]=inf.readLong(); c[i]=inf.readLong(); w[i]=inf.readLong();
        B += w[i];
    }

    // ---- read participant tour ----
    int L = ouf.readInt(0, 2*P, "L");
    vector<char> picked(P+1,0), delivered(P+1,0);
    long long load = 0;
    long long fuel = 0;
    long long curx = x0, cury = y0;

    for(int j=0;j<L;j++){
        int t = ouf.readInt(0,1,"t");
        int i = ouf.readInt(1,P,"i");
        long long nx,ny,dl;
        if(t==0){
            if(picked[i]) quitf(_wa,"contract %d picked up twice", i);
            picked[i]=1;
            nx=ax[i]; ny=ay[i]; dl=+m[i];
        } else {
            if(!picked[i]) quitf(_wa,"contract %d delivered before pickup", i);
            if(delivered[i]) quitf(_wa,"contract %d delivered twice", i);
            delivered[i]=1;
            nx=bx[i]; ny=by[i]; dl=-m[i];
        }
        // fly the leg into the new point at the CURRENT load
        long long d = manh(curx,cury,nx,ny);
        fuel += d*(K+load);
        // apply the event at the point
        load += dl;
        if(load > Q) quitf(_wa,"cargo capacity exceeded at event %d: load %lld > Q %lld", j+1, load, Q);
        if(load < 0) quitf(_wa,"negative cargo load at event %d", j+1);
        curx=nx; cury=ny;
    }
    if(!ouf.seekEof()) quitf(_wa,"trailing data after tour");

    // every performed contract must be delivered (no half-open contract)
    if(load != 0) quitf(_wa,"cargo hold not empty at return: a contract was picked up but never delivered");
    for(int i=1;i<=P;i++){
        if(picked[i] && !delivered[i]) quitf(_wa,"contract %d picked up but not delivered", i);
        if(delivered[i] && !picked[i]) quitf(_wa,"contract %d delivered but not picked up", i); // defensive
    }

    // final leg back to depot at load 0
    fuel += manh(curx,cury,x0,y0)*(K+load);

    // objective F = fuel + handling(performed) + penalty(omitted)
    long long F = fuel;
    for(int i=1;i<=P;i++){
        if(picked[i] && delivered[i]) F += c[i];
        else F += w[i];
    }

    if(B <= 0) B = 1; // safety; w_i>=1 guarantees B>=P>=1 anyway

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc/1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc/1000.0);
    return 0;
}
