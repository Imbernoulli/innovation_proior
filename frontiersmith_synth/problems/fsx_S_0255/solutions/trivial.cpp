// TIER: trivial
// Reproduces the checker's baseline exactly: type-0 anchored greedy, blind to value & depth.
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;
typedef vector<pii> Shape;

static Shape normalize(Shape s){int mnx=INT_MAX,mny=INT_MAX;for(auto&p:s){mnx=min(mnx,p.first);mny=min(mny,p.second);}for(auto&p:s){p.first-=mnx;p.second-=mny;}sort(s.begin(),s.end());return s;}
static string canon(const Shape&s0){Shape s=normalize(s0);string r;for(auto&p:s){r+=to_string(p.first);r+=',';r+=to_string(p.second);r+=';';}return r;}
static vector<Shape> orientations(const Shape&base){set<string> seen;vector<pair<string,Shape>> tmp;for(int refl=0;refl<2;refl++){Shape s=base;if(refl)for(auto&p:s)p.first=-p.first;for(int rot=0;rot<4;rot++){Shape t=s;for(int r=0;r<rot;r++)for(auto&p:t){int x=p.first,y=p.second;p.first=-y;p.second=x;}Shape nt=normalize(t);string c=canon(nt);if(!seen.count(c)){seen.insert(c);tmp.push_back({c,nt});}}}sort(tmp.begin(),tmp.end(),[](const pair<string,Shape>&a,const pair<string,Shape>&b){return a.first<b.first;});vector<Shape> out;for(auto&pr:tmp)out.push_back(pr.second);return out;}
static pii anchorOf(const Shape&s){int by=INT_MAX,bx=INT_MAX;for(auto&p:s)if(p.second<by||(p.second==by&&p.first<bx)){by=p.second;bx=p.first;}return{bx,by};}

int main(){
    int W,H,P; scanf("%d %d",&W,&H); scanf("%d",&P);
    vector<int> wval(P),rreq(P),stock(P),sz(P); vector<vector<Shape>> oris(P);
    for(int t=0;t<P;t++){scanf("%d %d %d %d",&wval[t],&rreq[t],&stock[t],&sz[t]);Shape base;for(int i=0;i<sz[t];i++){int dx,dy;scanf("%d %d",&dx,&dy);base.push_back({dx,dy});}oris[t]=orientations(base);}
    vector<vector<int>> D(H,vector<int>(W));
    for(int y=0;y<H;y++)for(int x=0;x<W;x++)scanf("%d",&D[y][x]);

    vector<vector<char>> cov(H,vector<char>(W,0));
    vector<pair<int,Shape>> placed;
    int used=0; bool stop=false;
    for(int y=0;y<H&&!stop;y++)for(int x=0;x<W&&!stop;x++){
        if(D[y][x]<rreq[0]||cov[y][x])continue;
        if(used>=stock[0]){stop=true;break;}
        for(auto&o:oris[0]){
            pii a=anchorOf(o); int ox=x-a.first,oy=y-a.second; bool fit=true;
            for(auto&p:o){int cx=ox+p.first,cy=oy+p.second;if(cx<0||cx>=W||cy<0||cy>=H||D[cy][cx]<rreq[0]||cov[cy][cx]){fit=false;break;}}
            if(fit){Shape ab;for(auto&p:o){int cx=ox+p.first,cy=oy+p.second;cov[cy][cx]=1;ab.push_back({cx,cy});}placed.push_back({0,ab});used++;break;}
        }
    }
    printf("%d\n",(int)placed.size());
    for(auto&pr:placed){printf("%d",pr.first);for(auto&c:pr.second)printf(" %d %d",c.first,c.second);printf("\n");}
    return 0;
}
