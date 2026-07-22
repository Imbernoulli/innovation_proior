# TIER: invalid
# Deliberately malformed answer (wrong shape) -- must score 0 under strict validation.
import sys, json

def main():
    json.load(sys.stdin)
    print(json.dumps({"assignment": "not-a-list"}))

if __name__ == "__main__":
    main()
