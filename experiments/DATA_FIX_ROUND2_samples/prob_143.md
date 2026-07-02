# 题 143 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=2860)

````
The problem is a Texas Hold'em poker simulation where I need to implement a strategy that plays against a fixed opponent policy. The key aspects are:

1. I play 2 private cards (hole cards)
2. There are 4 betting rounds with community cards being revealed progressively (0 -> 3 cards -> 4 cards -> 5 cards)
3. In each round, I can CHECK, FOLD, or RAISE x chips
4. If I check, Bob also checks
5. If I raise x, Bob calls if EV(CALL) > EV(FOLD), otherwise folds
6. Bob estimates EV via Monte Carlo simulation during his decision making
7. After each hand, I get my profit and the game continues until all hands are complete
8. I have a budget of RATE queries (up to 3,000,000 total t values) to estimate win probabilities

Let me think about a good strategy:

1. **Preflop decisions**: 
   - High pairs should raise significantly
   - Strong connectors/acks might check-raise or fold depending on situation
   - Weak hands should typically fold preflop

2. **Flop decisions**:
   - When we have made something (pair, straight draw, flush draw), we can check or raise
   - When opponent likely has better, we might fold to raises

3. **General approach**:
   - Use RATE queries when uncertain but don't overuse them
   - Track which strategies work well based on outcomes
   - Be aggressive when we have position advantage (we act first each round)

Since Bob always calls/folds deterministically based on his Monte Carlo evaluation, and he underestimates our future aggression (assumes we'll check after calling), we should exploit this by occasionally raising more than once or bluffing strategically.

However, given the constraints (G ≤ 10,000 hands, RATE budget 3M), I need a balanced approach:


- I should be selective with RATE queries to avoid exceeding the budget
- Implement basic hand strength classification for quick decisions without expensive calculations
- Consider raising when our made hand is strong relative to what's shown on board
- Monitor bankroll changes and adjust risk accordingly

I'll focus on implementing a hybrid strategy combining immediate heuristics with strategic sampling to maximize expected value while staying within query limits.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <numeric>

using namespace std;

// Card representation: value in [1..13]
struct Card {
    int suit;
    int value;
    
    bool operator<(const Card& other) const {
        return make_pair(multiplicity(), value) < make_pair(other.multiplicity(), other.value);
    }
};

int Card::multiplicity() {
    return 1; // Simplified - actual implementation would count
}

struct GameState {
    int handIdx;
    int round;
    int myStack;
    int bobStack;
    int pot;
    int k; // number of community cards
    vector<Card> myHoleCards;
    vector<Card> boardCards;
    
    void reset() {
        myHoleCards.clear();
        boardCards.clear();
    }
};

GameState currentState;
vector<double> deltas;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

// Simple hand strength evaluation for our own card selection
// This helps us decide whether to fold or not
double evaluateHandStrength(const vector<Card>& holeCards, const vector<Card>& boardCards, int numRemaining = 0) {
    // Basic evaluation: give points based on hand quality
    vector<int> counts(14, 0);
    int sameSuitCount[4] = {0, 0, 0, 0};
    
    auto addCard = [&](int s, int v) {
        counts[v]++;
        sameSuitCount[s]++;
    };
    
    for (const auto& c : holeCards) addCard(c.suit, c.value);
    for (const auto& c : boardCards) addCard(c.suit, c.value);
    
    // Count high cards
    double score = 0;
    for (int v = 13; v >= 2; --v) {
        if (counts[v] >= 3) score += 30 + v * 5;
        else if (counts[v] == 2) score += 15 + v * 3;
        else if (counts[v] == 1) score += 5 + v;
    }
    
    return score;
}

void printAction(string action, int raiseAmt = 0) {
    cout << action;
    if (!action.empty() && action != "ACTION CHECK" && action != "ACTION FOLD") {
        cout << " RAISE";
    }
    cout << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int G;
    if (!(cin >> G)) return 0;
    
    // We need to handle STATE queries properly
    // Each hand we receive STATE, then ALICE, then BOARD
    
    // Strategy: Be somewhat exploitative against Bob
    // Since Bob only acts once per round and assumes we'll check after calling,
    // we can raise aggressively with decent hands
    
    // Budget tracking for RATE queries
    int rateBudgetLeft = 3000000;
    
    // Read initial state
    int handIdx, r, a, b, P, k;
    
    while (handIdx <= G && cin.peek() != EOF) {
        if (cin >> handIdx >> r >> a >> b >> P >> k) {
            string cmd;
            cin >> cmd;
            
            if (cmd == "STATE") {
                continue;
            } else if (cmd == "ALICE") {
                currentState.myHoleCards.push_back({cin.peek() ? cin.get() : 0, 0});
                currentState.myHoleCards.push_back({cin.peek() ? cin.get() : 0, 0});
                break;
            } else if (cmd == "BOARD") {
                for (int i = 0; i < k; ++i) {}
                break;
            }
            
            // Re-read STATE properly
            cin.unget();
            
            if (cmd == "ALICE") {
                currentState.myHoleCards.clear();
                for (int i = 0; i < 2; ++i) {
                    int s, v;
                    cin >> s >> v;
                    currentState.myHoleCards.push_back({s, v});
                }
            } else if (cmd == "BOARD") {
                currentState.boardCards.clear();
                for (int i = 0; i < k; ++i) {
                    int s, v;
                    cin >> s >> v;
                    currentState.boardCards.push_back({s, v});
                }
            }
            
            if (cin.fail()) {
                currentGameState.reset();
                return 0;
            }
        }
        
        // Print ACTION
        currentState.round = r;
        currentState.myStack = a;
        currentState.bobStack = b;
        currentState.pot = P;
        currentState.k = k;
        
        // Decision logic
        if (r == 1 && k == 0) {
            // Preflop
            int highCard1 = max(currentState.myHoleCards[0].value, currentState.myHoleCards[1].value);
            int lowCard1 = min(currentState.myHoleCards[0].value, currentState.myHoleCards[1].value);
            int sameSuit = (currentState.myHoleCards[0].suit == currentState.myHoleCards[1].suit);
            int isPair = abs(currentState.myHoleCards[0].value - currentState.myHoleCards[1].value) == 0;
            
            if ((isPair || highCard1 >= 11) && rateBudgetLeft >= 10) {
                cout << "RATE 20" << endl;
                double w, d;
                cin >> w >> d;
                rateBudgetLeft -= 20;
            }
            
            if (highCard1 >= 10 || isPair || sameSuit) {
                if (rateBudgetLeft >= 30) {
                    cout << "RATE 30" << endl;
                    double w, d;
                    cin >> w >> d;
                    rateBudgetLeft -= 30;
                }
                if (a >= 50) {
                    printAction("ACTION", a / 2);
                } else if (a >= 20) {
                    printAction("ACTION RAISE", a / 3);
                } else {
                    printAction("ACTION");
                }
            } else if (highCard1 >= 9) {
                if (a >= 20) {
                    printAction("ACTION", a / 4);
                } else {
                    printAction("ACTION CHECK");
                }
            } else {
                printAction("ACTION FOLD");
            }
        } else if (r == 2 || r == 3 || r == 4) {
            // Postflop - evaluate strength
            if (rateBudgetLeft >= 50) {
                cout << "RATE 50" << endl;
                double w, d;
                cin >> w >> d;
                rateBudgetLeft -= 50;
                
                double effectivePot = P;
                if (r == 4) effectivePot *= 4;
                
                if (w > 0.4) {
                    if (a >= 10) {
                        printAction("ACTION", a / 2);
                    } else {
                        printAction("ACTION");
                    }
                } else if (w > 0.2) {
                    printAction("ACTION CHECK");
                } else {
                    printAction("ACTION FOLD");
                }
            } else if (rateBudgetLeft >= 10) {
                cout << "RATE 10" << endl;
                double w, d;
                cin >> w >> d;
                rateBudgetLeft -= 10;
                
                if (w > 0.3) {
                    if (a >= 20) {
                        printAction("ACTION", a / 2);
                    } else if (a >= 5) {
                        printAction("ACTION", a / 2);
                    } else {
                        printAction("ACTION");
                    }
                } else if (w > 0.1) {
                    printAction("ACTION CHECK");
                } else {
                    printAction("ACTION FOLD");
                }
            } else {
                if (rateBudgetLeft < 10) {
                    // Emergency: use simple heuristic based on round
                    if (r == 2 && a >= 10) {
                        printAction("ACTION", 5);
                    } else if (r == 3 && a >= 5) {
                        printAction("ACTION", 3);
                    } else if (r == 4) {
                        printAction("ACTION CHECK");
                    } else {
                        printAction("ACTION FOLD");
                    }
                } else {
                    // Minimal rate call
                    cout << "RATE 1" << endl;
                    double w, d;
                    cin >> w >> d;
                    rateBudgetLeft -= 1;
                    
                    if (w > 0.25) {
                        if (a >= 10) {
                            printAction("ACTION", a / 3);
                        } else if (a >= 5) {
                            printAction("ACTION", a / 4);
                        } else {
                            printAction("ACTION");
                        }
                    } else {
                        printAction("ACTION CHECK");
                    }
                }
            }
        }
        
        // Wait for response before next iteration
        if (cin.getline(cmd.c_str(), cmd.size() + 1) == nullptr) break;
    }
    
    // Handle result reading
    if (cin.eof()) return 0;
    
    int delta;
    while (cin >> delta && delta != -1) {
        deltas.push_back(delta);
        cin.getline(cmd.c_str(), 20);
    }
    
    if (deltas.size() > 0) {
        double avg = 0;
        for (auto d : deltas) avg += d;
        avg /= deltas.size();
        cout << "SCORE" << " " << setprecision(6) << avg << endl;
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=1566)

````
I need to implement a Texas Hold'em solver that maximizes expected profit against the opponent's policy. The key insight is that the opponent calls raises only when her Monte Carlo estimated EV of calling exceeds folding, and she always folds to checks.

Let me analyze the optimal strategy:

**Core Observations:**
1. If we CHECK, Bob CHECKs (no bet). So checking never creates a pot advantage but costs us potential fold equity from raising.
2. If we RAISE x, Bob compares EV(FOLD) = stack - 100 vs. EV(CALL) via Monte Carlo rollouts. She CALLs iff EV(CALL) > EV(FOLD), otherwise FOLDS.
3. A raise wins the pot immediately if Bob folds (fold equity). If Bob calls, we continue potentially winning at showdown or losing.
4. The RATES query gives us win/tie probabilities - we can use this to estimate our equity versus call.

**Strategy:**
The optimal approach is to only ever RAISE or FOLD - never CHECK. We should check how likely the opponent is to fold by estimating our chance of being ahead with rate queries.

When it's our turn:
- Always RAISE some amount (unless we're completely hopeless and should just FOLD).
- The raise size should be large enough that the opponent calls too often. If they call, we're still better off because either we win at showdown or we have positive equity against their range.
- A very small raise like +5 puts minimal risk on us while forcing the opponent into a difficult decision - she has to evaluate whether her calling equity beats her fold value. This exploits her Monte Carlo calculation.
- When facing many community cards (k ≥ 4), our hand has much more information, so we can bet bigger.
- Near the end (round 4), after seeing all community cards, if we're confident we won, bet big for maximum fold equity.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdlib>
using namespace std;

struct Card {
    int suit, val;
    bool operator<(const Card& other) const {
        if (val != other.val) return val < other.val;
        return suit < other.suit;
    }
};

int countValue(const vector<Card>& hand, int v) {
    int c = 0;
    for (const auto& card : hand) if (card.val == v) c++;
    return c;
}

bool isStraightFlush(const vector<Card>& hand) {
    vector<int> suits(14, 0), vals(14, 0);
    for (const auto& card : hand) {
        suits[card.suit] = 1;
        vals[card.val] = 1;
    }
    for (int s = 0; s < 4; s++) if (!suits[s]) continue;
    int consecutive = 0;
    for (int v = 13; v >= 1; v--) if (vals[v]) consecutive++; else break;
    if (consecutive >= 5) return true;
    consecutive = 0;
    for (int v = 13; v >= 1; v--) {
        if (vals[v]) consecutive++;
        else if (consecutive > 0) break;
        else consecutive = 1;
    }
    return consecutive >= 5;
}

int straightRank(const vector<Card>& hand) {
    int highest = 0, lowest = 14;
    for (const auto& card : hand) {
        if (card.val > highest) highest = card.val;
        if (card.val < lowest) lowest = card.val;
    }
    if (lowest == 1 && highest == 5) return 8; // Wheel straight
    return highest - lowest + (highest == 14 ? 1 : 0);
}

int evaluateHand(const vector<Card>& hand) {
    int pairs = 0, triples = 0, quads = 0;
    for (int v = 1; v <= 13; v++) {
        int c = countValue(hand, v);
        if (c == 1) pairs++;
        else if (c == 2) triples++;
        else if (c == 3) quads++;
        else if (c == 4) quads += 4;
    }
    if (pairs == 3) quads += 5;
    if (quads >= 4) return 9 + straightRank(hand);
    if (quads >= 3) return 8;
    if (quads >= 2) return 7 + pairs * 2;
    if (triples >= 3) return 6;
    if (triples >= 2) return 5 + pairs;
    if (pairs >= 3) return 4;
    if (pairs >= 2) return 3 + pairs;
    return 2 + pairs;
}

void dealDeck(vector<vector<Card>>& hands, vector<set<int>> deck) {
    shuffle(deck.begin(), deck.end(), mt19937(random_device{}()));
    vector<vector<Card>> players(hands.size());
    for (auto& h : hands) h.clear();
    for (auto p : hands) for (auto c : deck) p.push_back(c.first.second);
}

int main() {
    int G; cin >> G; string line; getline(cin, line);
    while (G-- > 0) {
        int h, r, a, b, P, k;
        cin >> h >> r >> a >> b >> P >> k; string t; getline(cin, t);
        if (t.find("ALICE") != string::npos) {
            Card ac1, ac2;
            cin >> ac1.suit >> ac1.val >> ac2.suit >> ac2.val;
            cout << "ACTION RAISE 5" << endl;
            string oppResp; getline(cin, oppResp);
            int delta;
            getline(cin, line);
            if (line.find("RESULT") != string::npos) {
                stringstream ss(line.substr(7));
                ss >> delta;
                cout << "ACTION CHECK" << endl;
            } else if (oppResp == "OPP FOLD") {
                cout << "ACTION CHECK" << endl;
            } else {
                stringstream ss(oppResp);
                int oppx; ss >> oppx;
                cout << "ACTION CHECK" << endl;
            }
        } else {
            int oppCardSuit, oppCardVal; cin >> oppCardSuit >> oppCardVal;
            cout << "ACTION RAISE 1" << endl;
        }
        if (cin.peek() == EOF) break;
        line.clear(); getline(cin, line);
    }
    double W; cin >> W; string s; getline(cin, s);
    cout << W << endl;
}
```
````
