// TIER: strong
// Density local search: seeded multi-restart randomized cell orders; at each empty cell try
// EVERY sub-cell of EVERY orientation of EVERY in-stock type as the cell landing there
// (respecting the shielding threshold), select the legal placement of highest value-per-area
// w_t / s_t so scarce high-value deep bays get packed onto deep-regolith patches. Keep the
// layout of maximum total value across all restarts.
#include <bits/stdc++.h>
using namespace std;
typedef pair<int,int> pii;
typedef vector<pii> Shape;

static Shape normalize(Shape s){int mnx=INT_MAX,mny=INT_MAX;for(auto&p:s){mnx=min(mnx,p.first);mny=min(mny,p.second);}for(auto&p:s){p.first-=mnx;p.second-=mny;}sort(s.begin(),s.end());return s;}
static string canon(const Shape&s0){Shape s=normalize(s0);string r;for(auto&p:s){r+=to_string(p.first);r+=',';r+=to_string(p.second);r+=';';}return r;}
static vector<Shape> orientations(const Shape&base){set<string> seen;vector<pair<string,Shape>> tmp;for(int refl=0;refl<2;refl++){Shape s=base;if(refl)for(auto&p:s)p.first=-p.first;for(int rot=0;rot<4;rot++){Shape t=s;for(int r=0;r<rot;r++)for(auto&p:t){int x=p.first,y=p.second;p.first=-y;p.second=x;}Shape nt=normalize(t);string c=canon(nt);if(!seen.count(c)){seen.insert(c);tmp.push_back({c,nt});}}}sort(tmp.begin(),tmp.end(),[](const pair<string,Shape>&a,const pair<string,Shape>&b){return a.first<b.first;});vector<Shape> out;for(auto&pr:tmp)out.push_back(pr.second);return out;}

static uint64_t rngState = 0x9e3779b97f4a7c15ULL;
static inline uint64_t nextRand(){uint64_t z=(rngState+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}

int W,H,P;
vector<int> wval,rreq,stock,sz; vector<vector<Shape>> oris; vector<vector<int>> D;

int main(){
    scanf("%d %d",&W,&H); scanf("%d",&P);
    wval.resize(P);rreq.resize(P);stock.resize(P);sz.resize(P);oris.resize(P);
    for(int t=0;t<P;t++){scanf("%d %d %d %d",&wval[t],&rreq[t],&stock[t],&sz[t]);Shape base;for(int i=0;i<sz[t];i++){int dx,dy;scanf("%d %d",&dx,&dy);base.push_back({dx,dy});}oris[t]=orientations(base);}
    D.assign(H,vector<int>(W));
    for(int y=0;y<H;y++)for(int x=0;x<W;x++)scanf("%d",&D[y][x]);

    vector<int> order(W*H); for(int i=0;i<W*H;i++)order[i]=i;
    int restarts = max(6, min(24, 200000/max(1,W*H)));
    long long bestScore=-1; vector<pair<int,Shape>> bestPlaced;

    for(int rs=0; rs<restarts; rs++){
        for(int i=W*H-1;i>0;i--){int j=(int)(nextRand()%(uint64_t)(i+1));swap(order[i],order[j]);}
        vector<vector<char>> cov(H,vector<char>(W,0));
        vector<int> cnt(P,0);
        vector<pair<int,Shape>> placed;
        long long total=0;
        for(int idx=0; idx<W*H; idx++){
            int cell=order[idx]; int sx=cell%W, sy=cell/W;
            if(cov[sy][sx])continue;
            double bestDens=-1; long long bestW=-1; int bT=-1; Shape bCells;
            for(int t=0;t<P;t++){
                if(cnt[t]>=stock[t])continue;
                for(auto&o:oris[t]){
                    for(auto&anc:o){
                        int ox=sx-anc.first, oy=sy-anc.second; bool fit=true; Shape ab;
                        for(auto&p:o){int cx=ox+p.first,cy=oy+p.second;if(cx<0||cx>=W||cy<0||cy>=H||D[cy][cx]<rreq[t]||cov[cy][cx]){fit=false;break;}ab.push_back({cx,cy});}
                        if(fit){ double dens=(double)wval[t]/(double)sz[t];
                                 if(dens>bestDens+1e-9 || (fabs(dens-bestDens)<=1e-9 && wval[t]>bestW)){bestDens=dens;bestW=wval[t];bT=t;bCells=ab;} }
                    }
                }
            }
            if(bT>=0){for(auto&c:bCells)cov[c.second][c.first]=1;cnt[bT]++;placed.push_back({bT,bCells});total+=wval[bT];}
        }
        if(total>bestScore){bestScore=total;bestPlaced=placed;}
    }

    printf("%d\n",(int)bestPlaced.size());
    for(auto&pr:bestPlaced){printf("%d",pr.first);for(auto&c:pr.second)printf(" %d %d",c.first,c.second);printf("\n");}
    return 0;
}
