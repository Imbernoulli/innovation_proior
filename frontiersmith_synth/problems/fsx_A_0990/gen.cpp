#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- "One Dictionary for Legal Contracts, Chat Logs, and Code".
// Builds K=3 domain corpora (legal, chat, code) from a SHARED pool of NSTEMS hidden
// "stems" combined with DOMAIN-EXCLUSIVE prefix/suffix alphabets (disjoint character
// ranges per domain, plus a disjoint stem alphabet), so a substring drawn from a domain's
// own affix range can literally never occur in another domain's corpus, while stem-only
// substrings genuinely recur across all three. legal has few, narrow affixes and heavily
// favors a small CORE subset of stems (near-boilerplate, extreme repetition of whole
// words); chat is moderately skewed; code samples the full stem pool uniformly with a
// wide, diverse affix alphabet. legal is also made far larger than chat, which is larger
// than code -- pooled-frequency counting over the concatenation is then dominated by
// legal's own repeated whole words, starving code's own vocabulary of dictionary slots
// (the intended trap for a domain-blind greedy). testId is a difficulty ladder: 1 tiny
// (example-scale), growing to the full envelope at 10.

int NSTEMS, CORESZ; long long DMAX;
long long RLEG, RCHAT, RCODE;

void setparams(int t){
    switch(t){
        case 1:  NSTEMS=8;  DMAX=10; RLEG=300;   RCHAT=200;   RCODE=100;  break; // tiny
        case 2:  NSTEMS=10; DMAX=12; RLEG=800;   RCHAT=500;   RCODE=250;  break;
        case 3:  NSTEMS=12; DMAX=16; RLEG=1800;  RCHAT=1100;  RCODE=500;  break;
        case 4:  NSTEMS=16; DMAX=20; RLEG=6000;  RCHAT=2500;  RCODE=800;  break;  // trap
        case 5:  NSTEMS=18; DMAX=24; RLEG=9000;  RCHAT=3500;  RCODE=1000; break;  // trap
        case 6:  NSTEMS=20; DMAX=28; RLEG=13000; RCHAT=5000;  RCODE=1300; break;  // trap
        case 7:  NSTEMS=22; DMAX=32; RLEG=18000; RCHAT=7000;  RCODE=1600; break;  // trap
        case 8:  NSTEMS=24; DMAX=36; RLEG=24000; RCHAT=9000;  RCODE=2000; break;  // trap
        case 9:  NSTEMS=26; DMAX=40; RLEG=30000; RCHAT=11000; RCODE=2400; break;  // trap
        default: NSTEMS=28; DMAX=48; RLEG=40000; RCHAT=14000; RCODE=3000; break;  // full envelope, trap
    }
    CORESZ = max(2, NSTEMS/4);
}

string randStr(int len, const string& alpha){
    string s;
    s.reserve(len);
    for(int i=0;i<len;i++) s += alpha[rnd.next((int)alpha.size())];
    return s;
}

vector<string> genAffixes(int cnt, int len, const string& alpha){
    set<string> seen;
    vector<string> out;
    int guard=0;
    while((int)out.size() < cnt && guard < 20000){
        guard++;
        string s = randStr(len, alpha);
        if(seen.insert(s).second) out.push_back(s);
    }
    return out;
}

int main(int argc, char** argv){
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    setparams(t);

    // ---- shared stem pool (disjoint alphabet "uvwxyz") ----
    const string STEM_ALPHA = "uvwxyz";
    set<string> stemset;
    vector<string> stems;
    while((int)stems.size() < NSTEMS){
        int len = 4 + rnd.next(3); // 4..6
        string s = randStr(len, STEM_ALPHA);
        if(stemset.insert(s).second) stems.push_back(s);
    }

    // ---- domain-exclusive affix alphabets (disjoint from stems and from each other) ----
    vector<string> legalP = genAffixes(2, 2, "ab");
    vector<string> legalS = genAffixes(2, 2, "ab");
    vector<string> chatP  = genAffixes(4, 2, "cd");
    vector<string> chatS  = genAffixes(4, 2, "cd");
    vector<string> codeP  = genAffixes(8, 3, "efghijklmnopqrst");
    vector<string> codeS  = genAffixes(8, 3, "efghijklmnopqrst");

    auto buildCorpus = [&](long long targetLen, vector<string>& P, vector<string>& S,
                            double coreProb) -> string {
        string out;
        out.reserve(targetLen + 32);
        int coreThresh = (int)llround(coreProb * 100.0);
        while((long long)out.size() < targetLen){
            int stemIdx;
            if(coreThresh > 0 && rnd.next(100) < coreThresh) stemIdx = rnd.next(CORESZ);
            else stemIdx = rnd.next(NSTEMS);
            const string& stem = stems[stemIdx];
            const string& pre  = P[rnd.next((int)P.size())];
            const string& suf  = S[rnd.next((int)S.size())];
            out += pre; out += stem; out += suf;
        }
        out.resize((size_t)targetLen);
        return out;
    };

    string legal = buildCorpus(RLEG,  legalP, legalS, 0.75);
    string chat  = buildCorpus(RCHAT, chatP,  chatS,  0.40);
    string code  = buildCorpus(RCODE, codeP,  codeS,  0.0);

    printf("%d %lld\n", 3, DMAX);
    printf("%lld\n%s\n", (long long)legal.size(), legal.c_str());
    printf("%lld\n%s\n", (long long)chat.size(),  chat.c_str());
    printf("%lld\n%s\n", (long long)code.size(),  code.c_str());
    return 0;
}
