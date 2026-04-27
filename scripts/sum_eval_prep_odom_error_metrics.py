import sys
import os
import json
sys.path.append("../")

from odometry.analyzers.analyzer import Analyzer

# Configuration lookup
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "analyzer_configs",
    "analyzer_config.json"
)

#desired results folder and file path
SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sum_eval_odom_error_comparison",
    "odom_error_comparison_summary.csv"
)

def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
        
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Initialize the analyzer
    analyzer = Analyzer()

    print(f"Generating Odometry Error Comparison Table from {CONFIG_PATH}...")
    try:
        analyzer.generate_odom_error_comparison_table(
            config=config,
            percentile=0.90,
            save_path=SAVE_PATH
        )
    except FileNotFoundError as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
