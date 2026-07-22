# TIER: invalid
import sys, json

def main():
    _ = json.load(sys.stdin)
    # out-of-range level -> must be rejected by the checker
    print(json.dumps({"level": 99}))

main()
