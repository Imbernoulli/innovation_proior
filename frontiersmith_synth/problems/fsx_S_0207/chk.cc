#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int H, W, T;
vector<vector<int>> val;                       // grid values
vector<vector<pair<int,int>>> shape;           // base normalized shapes
vector<int> cnt;

// normalized cells of shape i under orientation o (0..7)
vector<pair<int,int>> oriented(int i, int o){
    int reflect = o/4, rot = o%4;
    vector<pair<int,int>> cs;
    for(auto&p:shape[i]){
        int x=p.first, y=p.second;
        if(reflect) y=-y;
        for(int k=0;k<rot;k++){ int nx=y, ny=-x; x=nx; y=ny; }
        cs.push_back({x,y});
    }
    int mr=INT_MAX, mc=INT_MAX;
    for(auto&p:cs){mr=min(mr,p.first); mc=min(mc,p.second);}
    for(auto&p:cs){p.first-=mr; p.second-=mc;}
    sort(cs.begin(),cs.end());
    return cs;
}

int main(int argc,char*argv[]){
    registerTestlibCmd(argc,argv);

    H=inf.readInt(); W=inf.readInt(); T=inf.readInt();
    val.assign(H, vector<int>(W,0));
    for(int r=0;r<H;r++)for(int c=0;c<W;c++) val[r][c]=inf.readInt();
    shape.resize(T); cnt.resize(T);
    long long totalSupply=0;
    for(int i=0;i<T;i++){
        cnt[i]=inf.readInt();
        int s=inf.readInt();
        for(int k=0;k<s;k++){int dr=inf.readInt(), dc=inf.readInt(); shape[i].push_back({dr,dc});}
        totalSupply += cnt[i];
    }

    // ---- internal baseline B: reading-order fill with type 0, orientation 0 ----
    long long B=0;
    {
        vector<vector<char>> occ(H, vector<char>(W,0));
        auto cs = oriented(0,0);
        int left=cnt[0];
        for(int r=0;r<H && left>0;r++){
            for(int c=0;c<W && left>0;c++){
                bool fit=true;
                for(auto&p:cs){int ar=r+p.first, ac=c+p.second; if(ar<0||ar>=H||ac<0||ac>=W||occ[ar][ac]){fit=false;break;}}
                if(!fit) continue;
                for(auto&p:cs){int ar=r+p.first, ac=c+p.second; occ[ar][ac]=1; B+=val[ar][ac];}
                left--;
            }
        }
    }
    if(B<=0) B=1;

    // ---- read + validate participant output ----
    long long K = ouf.readLong(0, totalSupply, "K");
    vector<vector<char>> occ(H, vector<char>(W,0));
    vector<int> used(T,0);
    long long F=0;
    for(long long k=0;k<K;k++){
        int i=ouf.readInt(0,T-1,"type");
        int o=ouf.readInt(0,7,"orient");
        int r=ouf.readInt(0,H-1,"r");
        int c=ouf.readInt(0,W-1,"c");
        used[i]++;
        if(used[i]>cnt[i]) quitf(_wa,"type %d used %d > cnt %d", i, used[i], cnt[i]);
        auto cs = oriented(i,o);
        // validate all cells first
        for(auto&p:cs){
            int ar=r+p.first, ac=c+p.second;
            if(ar<0||ar>=H||ac<0||ac>=W) quitf(_wa,"placement %lld out of bounds at (%d,%d)", k, ar, ac);
            if(occ[ar][ac]) quitf(_wa,"placement %lld overlaps at (%d,%d)", k, ar, ac);
        }
        for(auto&p:cs){int ar=r+p.first, ac=c+p.second; occ[ar][ac]=1; F+=val[ar][ac];}
    }
    if(!ouf.seekEof()) quitf(_wa,"trailing output");

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL,B));
    quitp(sc/1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc/1000.0);
    return 0;
}
