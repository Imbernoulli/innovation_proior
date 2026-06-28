#include <bits/stdc++.h>
using namespace std;
int H,W,K,N; vector<char> blk; vector<int> sr,sc,tr,tc;
inline int ID(int r,int c){return r*W+c;}
const int DR[4]={-1,1,0,0},DC[4]={0,0,-1,1};
int main(){ios::sync_with_stdio(false);cin.tie(0);
 if(!(cin>>H>>W>>K)){cout<<0<<"\n";return 0;} N=H*W;blk.assign(N,0);string row;
 for(int r=0;r<H;r++){cin>>row;for(int c=0;c<W;c++)blk[ID(r,c)]=(row[c]=='#');}
 sr.resize(K);sc.resize(K);tr.resize(K);tc.resize(K);
 for(int i=0;i<K;i++)cin>>sr[i]>>sc[i]>>tr[i]>>tc[i];
 vector<int> ord(K);iota(ord.begin(),ord.end(),0);
 sort(ord.begin(),ord.end(),[&](int a,int b){return abs(sr[a]-tr[a])+abs(sc[a]-tc[a])<abs(sr[b]-tr[b])+abs(sc[b]-tc[b]);});
 vector<char> cl(N,0);vector<vector<int>> P(K);vector<int> pr(N),gn(N,-1);int tk=0;
 for(int i:ord){int s=ID(sr[i],sc[i]),t=ID(tr[i],tc[i]);if(cl[s]||cl[t])continue;tk++;queue<int>q;q.push(s);gn[s]=tk;pr[s]=-1;bool f=0;
  while(!q.empty()){int u=q.front();q.pop();if(u==t){f=1;break;}int r=u/W,c=u%W;for(int d=0;d<4;d++){int nr=r+DR[d],nc=c+DC[d];if(nr<0||nr>=H||nc<0||nc>=W)continue;int v=nr*W+nc;if(blk[v]||cl[v]||gn[v]==tk)continue;gn[v]=tk;pr[v]=u;q.push(v);}}
  if(!f)continue;vector<int>p;for(int v=t;v!=-1;v=pr[v])p.push_back(v);for(int v:p)cl[v]=1;P[i]=p;}
 vector<int> idx;for(int i=0;i<K;i++)if(!P[i].empty())idx.push_back(i);
 cout<<idx.size()<<"\n";for(int i:idx){auto&p=P[i];cout<<i<<" "<<p.size();for(int v:p)cout<<" "<<v/W<<" "<<v%W;cout<<"\n";}return 0;}
