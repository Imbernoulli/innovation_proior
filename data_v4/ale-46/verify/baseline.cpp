// Trivial baseline: greedy shortest-path-per-pair (BFS) in the GIVEN pair order,
// claiming cells as it goes; a pair that cannot be routed on the remaining free
// cells is simply skipped.  Counts only conflict-free pairs.  This is the
// "greedy shortest-path-per-pair (counting only conflict-free pairs)"
// normalization baseline named in the scoring rule.
#include <bits/stdc++.h>
using namespace std;
int H, W, K, N;
vector<int> epA, epB, owner, prevc;
static inline int CR(int id){return id/W;} static inline int CC(int id){return id%W;}
int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    if(!(cin>>H>>W>>K)) return 0; N=H*W;
    epA.resize(K); epB.resize(K);
    for(int p=0;p<K;++p){int r1,c1,r2,c2;cin>>r1>>c1>>r2>>c2;epA[p]=r1*W+c1;epB[p]=r2*W+c2;}
    owner.assign(N,-1); prevc.assign(N,-1);
    // Pre-block every endpoint cell with a sentinel owner so a routed path never
    // steals another pair's endpoint; we temporarily free the current pair's own
    // endpoints while routing it.
    const int EP=-3;
    for(int p=0;p<K;++p){ owner[epA[p]]=EP; owner[epB[p]]=EP; }
    vector<vector<int>> path(K);
    static const int DR[4]={-1,1,0,0}, DC[4]={0,0,-1,1};
    for(int p=0;p<K;++p){
        int s=epA[p],t=epB[p];
        owner[s]=-1; owner[t]=-1; // own endpoints traversable
        fill(prevc.begin(),prevc.end(),-2);
        queue<int> q; q.push(s); prevc[s]=-1;
        bool found=false;
        while(!q.empty()){
            int u=q.front();q.pop();
            if(u==t){found=true;break;}
            int ur=u/W,uc=u%W;
            for(int d=0;d<4;++d){int nr=ur+DR[d],nc=uc+DC[d];
                if(nr<0||nr>=H||nc<0||nc>=W)continue;int v=nr*W+nc;
                if(prevc[v]!=-2)continue;
                if(v!=t && owner[v]!=-1)continue; // blocked
                prevc[v]=u;q.push(v);}
        }
        if(!found){ owner[s]=EP; owner[t]=EP; continue; } // restore endpoint blocks
        vector<int> pp; int cur=t; while(cur!=-1){pp.push_back(cur);cur=prevc[cur];}
        reverse(pp.begin(),pp.end());
        for(int c:pp) owner[c]=p; path[p]=pp;
    }
    vector<int> good; for(int p=0;p<K;++p) if(!path[p].empty()) good.push_back(p);
    cout<<good.size()<<"\n";
    for(int p:good){cout<<p<<" "<<path[p].size();for(int c:path[p])cout<<" "<<CR(c)<<" "<<CC(c);cout<<"\n";}
    return 0;
}
