#include <bits/stdc++.h>
using namespace std;
// Trivial baseline: minimum-carve connecting path (straight-ish corridor),
// then pad to exactly B with arbitrary remaining walls. Always feasible.
static int H,W,B,SR,SC,TR,TC,N;
static inline int id(int r,int c){return r*W+c;}
static const int DR[4]={1,-1,0,0},DC[4]={0,0,1,-1};
int main(){
    ios::sync_with_stdio(false);cin.tie(nullptr);
    if(!(cin>>H>>W>>B))return 0;
    cin>>SR>>SC>>TR>>TC;
    vector<string>G(H);
    for(int r=0;r<H;r++){string s;cin>>s;if((int)s.size()<W)s.resize(W,'.');G[r]=s;}
    N=H*W;
    vector<char>isWall(N,0);
    for(int r=0;r<H;r++)for(int c=0;c<W;c++)isWall[id(r,c)]=(G[r][c]=='#');
    isWall[id(SR,SC)]=0;isWall[id(TR,TC)]=0;
    int tw=0;for(int u=0;u<N;u++)tw+=isWall[u];
    if(B>tw)B=tw;
    // 0/1 BFS min-carve path
    vector<int>dist(N,INT_MAX),par(N,-1);
    deque<int>dq;int s=id(SR,SC),t=id(TR,TC);dist[s]=0;dq.push_back(s);
    while(!dq.empty()){int u=dq.front();dq.pop_front();int r=u/W,c=u%W;
        for(int k=0;k<4;k++){int nr=r+DR[k],nc=c+DC[k];if(nr<0||nr>=H||nc<0||nc>=W)continue;
            int v=id(nr,nc),w=isWall[v]?1:0;
            if(dist[u]+w<dist[v]){dist[v]=dist[u]+w;par[v]=u;if(w==0)dq.push_front(v);else dq.push_back(v);}}}
    vector<char>chosen(N,0);vector<int>carve;
    int u=t;while(u!=-1){if(isWall[u]&&!chosen[u]){chosen[u]=1;carve.push_back(u);}u=par[u];}
    // pad
    for(int v=0;v<N&&(int)carve.size()<B;v++)if(isWall[v]&&!chosen[v]){chosen[v]=1;carve.push_back(v);}
    if((int)carve.size()>B)carve.resize(B);
    string out;for(int x:carve){out+=to_string(x/W);out+=' ';out+=to_string(x%W);out+='\n';}
    cout<<out;return 0;
}
