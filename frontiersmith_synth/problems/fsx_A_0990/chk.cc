#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for "One Dictionary for Legal Contracts, Chat Logs, and Code" (MAXIMIZE).
// Participant lists D<=Dmax distinct lowercase dictionary entries, length in [3,12]. The
// FROZEN GREEDY LONGEST-MATCH PARSER (lengths 3..12 only) encodes each of the 3 domain
// corpora with the dictionary: a match costs 2 units, a literal byte costs 1 unit.
// F(S) = min over the 3 domains of (R_d - encoded_d(S)) / R_d  (worst-corpus savings ratio).
// Internal baseline B_set = the Dmax most frequent length-3 substrings in the pooled
// concatenation of all 3 corpora (ties lexicographic) -- a domain-blind, length-blind
// frequency count. Bb = F(B_set) (always > 0 by construction: see below).
// ratio = min(1, 0.075 * F(S) / Bb)  ->  matching the baseline dictionary scores 0.075.

static const int LMIN = 3, LMAX = 12, TOKEN_COST = 2;

// encode corpus T with dictionary set `dict` via the frozen greedy longest-match parser;
// return savings = R - encoded.
static long long parseSavings(const string& T, const unordered_set<string>& dict){
    int n = (int)T.size();
    long long encoded = 0;
    int i = 0;
    while(i < n){
        int maxLen = min(LMAX, n - i);
        int matched = -1;
        for(int L = maxLen; L >= LMIN; L--){
            if(dict.count(T.substr(i, L))){ matched = L; break; }
        }
        if(matched > 0){ encoded += TOKEN_COST; i += matched; }
        else{ encoded += 1; i += 1; }
    }
    return (long long)n - encoded;
}

static double worstRatio(const vector<string>& corpora, const vector<long long>& R,
                          const unordered_set<string>& dict){
    double worst = 1e18;
    for(size_t d=0; d<corpora.size(); d++){
        long long sav = parseSavings(corpora[d], dict);
        double ratio = (double)sav / (double)R[d];
        worst = min(worst, ratio);
    }
    return worst;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int K = inf.readInt();
    long long Dmax = inf.readLong();
    vector<string> corpora(K);
    vector<long long> R(K);
    for(int d=0; d<K; d++){
        R[d] = inf.readLong();
        corpora[d] = inf.readToken();
        if((long long)corpora[d].size() != R[d])
            quitf(_fail, "generator/input malformed: domain %d length mismatch", d);
    }

    // ---- build grader's internal baseline dictionary: top-Dmax length-3 substrings, pooled ----
    unordered_map<string,long long> pooled3;
    for(int d=0; d<K; d++){
        const string& T = corpora[d];
        int n = (int)T.size();
        for(int i=0; i+LMIN<=n; i++) pooled3[T.substr(i,LMIN)]++;
    }
    vector<pair<long long,string>> ranked;
    ranked.reserve(pooled3.size());
    for(auto& kv : pooled3) ranked.push_back({kv.second, kv.first});
    sort(ranked.begin(), ranked.end(), [](const pair<long long,string>& a, const pair<long long,string>& b){
        if(a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    unordered_set<string> baseDict;
    for(long long i=0; i < (long long)ranked.size() && i < Dmax; i++) baseDict.insert(ranked[i].second);

    double Bb = worstRatio(corpora, R, baseDict);
    if(!(Bb > 0.0)) Bb = 1e-9;

    // ---- read participant dictionary ----
    long long D = ouf.readLong(0, Dmax, "D");
    unordered_set<string> dict;
    dict.reserve((size_t)D*2+16);
    for(long long i=0; i<D; i++){
        string s = ouf.readToken();
        int len = (int)s.size();
        if(len < LMIN || len > LMAX)
            quitf(_wa, "entry %lld has length %d, must be in [%d,%d]", i+1, len, LMIN, LMAX);
        for(char c : s){
            if(c < 'a' || c > 'z')
                quitf(_wa, "entry %lld = \"%s\" contains a non-lowercase-letter character", i+1, s.c_str());
        }
        if(dict.count(s))
            quitf(_wa, "entry %lld = \"%s\" duplicates an earlier entry", i+1, s.c_str());
        dict.insert(s);
    }
    if(!ouf.seekEof()) quitf(_wa, "trailing output after the dictionary");

    double F = worstRatio(corpora, R, dict);
    if(!isfinite(F) || F < 0) F = 0.0;

    double sc = min(1000.0, 75.0 * F / max(1e-9, Bb));
    double ratio = sc / 1000.0;
    quitp(ratio, "OK F=%.8f Bb=%.8f D=%lld Ratio: %.6f", F, Bb, D, ratio);
    return 0;
}
