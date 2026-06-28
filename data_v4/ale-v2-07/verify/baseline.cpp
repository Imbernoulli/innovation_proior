// Trivial baseline: sequential A* in INPUT (arbitrary) order, hard-blocking claimed
// cells, skipping a net when it is boxed in. No rip-up, no congestion, no ordering.
#include <bits/stdc++.h>
using namespace std;
int H,W,K,N;
vector<char> blk;
vector<int> sr,sc,tr,tc;
inline int ID(int r,int c){return r*W+c;}
const int DR[4]={-1,1,0,0},DC[4]={0,0,-1,1};
int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    if(!(cin>>H>>W>>K)){cout<<0<<"\n";return 0;}
    N=H*W; blk.assign(N,0);
    string row;
    for(int r=0;r<H;r++){cin>>row; for(int c=0;c<W;c++) blk[ID(r,c)]=(row[c]=='#');}
    sr.resize(K);sc.resize(K);tr.resize(K);tc.resize(K);
    for(int i=0;i<K;i++) cin>>sr[i]>>sc[i]>>tr[i]>>tc[i];
    vector<char> claimed(N,0);
    vector<vector<int>> paths(K);
    vector<int> prev(N), gen(N,-1); int tok=0;
    for(int i=0;i<K;i++){
        int s=ID(sr[i],sc[i]),t=ID(tr[i],tc[i]);
        if(claimed[s]||claimed[t]) continue;
        // plain BFS (unit cost) avoiding claimed/blocked
        tok++;
        queue<int> q; q.push(s); gen[s]=tok; prev[s]=-1;
        bool found=false;
        while(!q.empty()){
            int u=q.front();q.pop();
            if(u==t){found=true;break;}
            int r=u/W,c=u%W;
            for(int d=0;d<4;d++){int nr=r+DR[d],nc=c+DC[d];
                if(nr<0||nr>=H||nc<0||nc>=W)continue;
                int v=nr*W+nc;
                if(blk[v]||claimed[v]||gen[v]==tok)continue;
                gen[v]=tok;prev[v]=u;q.push(v);}
        }
        if(!found)continue;
        vector<int> p; for(int v=t;v!=-1;v=prev[v])p.push_back(v);
        reverse(p.begin(),p.end());
        for(int v:p)claimed[v]=1; paths[i]=p;
    }
    vector<int> idx; for(int i=0;i<K;i++) if(!paths[i].empty()) idx.push_back(i);
    cout<<idx.size()<<"\n";
    for(int i:idx){auto&p=paths[i];cout<<i<<" "<<p.size();for(int v:p)cout<<" "<<v/W<<" "<<v%W;cout<<"\n";}
    return 0;
}
