#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int H, W;

// random connected polyomino of given size, normalized (min row/col = 0)
vector<pair<int,int>> genPoly(int sz){
    set<pair<int,int>> cells;
    cells.insert({0,0});
    int dr[]={0,0,1,-1}, dc[]={1,-1,0,0};
    int guard=0;
    while((int)cells.size()<sz && guard++<10000){
        vector<pair<int,int>> v(cells.begin(),cells.end());
        auto p=v[rnd.next(0,(int)v.size()-1)];
        int d=rnd.next(0,3);
        cells.insert({p.first+dr[d], p.second+dc[d]});
    }
    int mr=INT_MAX, mc=INT_MAX;
    for(auto&p:cells){mr=min(mr,p.first); mc=min(mc,p.second);}
    vector<pair<int,int>> res;
    for(auto&p:cells) res.push_back({p.first-mr, p.second-mc});
    sort(res.begin(),res.end());
    return res;
}

int main(int argc,char*argv[]){
    registerGen(argc,argv,1);
    int t=atoi(argv[1]);

    int n = (t<=1)?6 : min(6 + t*5, 56);
    H=n; W=n;
    int T = (t<=1)?2 : min(2 + t, 8);

    printf("%d %d %d\n", H, W, T);

    // conservation value field: gradient rising toward bottom-right + noise +
    // a couple of high-value clusters, so where you place matters a lot.
    vector<vector<int>> v(H, vector<int>(W,1));
    for(int r=0;r<H;r++)for(int c=0;c<W;c++){
        double g = 10.0 + 75.0 * (double)(r+c) / (double)max(1,(H+W-2));
        int noise = rnd.next(-5,5);
        int val = (int)llround(g) + noise;
        if(val<1) val=1; if(val>100) val=100;
        v[r][c]=val;
    }
    // a few bright clusters (placed anywhere, values near max)
    int clusters = 1 + t/3;
    for(int k=0;k<clusters;k++){
        int cr=rnd.next(0,H-1), cc=rnd.next(0,W-1);
        int rad=rnd.next(1,3);
        for(int r=max(0,cr-rad);r<=min(H-1,cr+rad);r++)
            for(int c=max(0,cc-rad);c<=min(W-1,cc+rad);c++)
                v[r][c]=max(v[r][c], rnd.next(85,100));
    }
    for(int r=0;r<H;r++){
        for(int c=0;c<W;c++) printf("%d%c", v[r][c], c+1==W?'\n':' ');
    }

    long long A = (long long)H*W;

    // build type shapes. type 0 = 2x2 square.
    vector<vector<pair<int,int>>> shapes;
    shapes.push_back({{0,0},{0,1},{1,0},{1,1}});
    for(int i=1;i<T;i++){
        int sz=rnd.next(2,5);
        vector<pair<int,int>> s;
        for(int tries=0;tries<50;tries++){
            s=genPoly(sz);
            int mh=0,mw=0; for(auto&p:s){mh=max(mh,p.first); mw=max(mw,p.second);}
            if(mh<H && mw<W) break;
        }
        shapes.push_back(s);
    }

    // counts: type 0 covers ~20% of area; the rest share ~35%.
    vector<int> cnt(T,1);
    cnt[0] = max(1, (int)llround(0.20*A/4.0));
    long long remCells = (long long)llround(0.35*A);
    for(int i=1;i<T;i++){
        int sz=(int)shapes[i].size();
        long long share = remCells/max(1,(T-1));
        cnt[i]=max(1, (int)llround((double)share/(double)sz));
    }
    // safety: keep total supply <= A
    long long tot=0; for(int i=0;i<T;i++) tot += (long long)cnt[i]*(long long)shapes[i].size();
    while(tot>A){
        // trim the largest contributor
        int bi=0; long long bv=-1;
        for(int i=0;i<T;i++){long long cv=(long long)cnt[i]*shapes[i].size(); if(cv>bv){bv=cv;bi=i;}}
        if(cnt[bi]<=1) break;
        tot -= shapes[bi].size(); cnt[bi]--;
    }

    for(int i=0;i<T;i++){
        printf("%d %d\n", cnt[i], (int)shapes[i].size());
        for(auto&p:shapes[i]) printf("%d %d\n", p.first, p.second);
    }
    return 0;
}
