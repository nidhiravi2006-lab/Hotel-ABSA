"""Command-line demo: python predict.py --text 'The room was dirty.'"""
import argparse
import json
from absa.core import Analyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--all-aspects", action="store_true", help="Also predict trained aspects with no keyword mention")
    args = parser.parse_args()
    try:
        print(json.dumps(Analyzer().analyze(args.text, args.all_aspects), indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, str(error) + "\n")
