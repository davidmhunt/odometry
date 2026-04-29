import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append("../")

from odometry.analyzers.analyzer import Analyzer

# Configuration lookup
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "analyzer_configs",
    "analyzer_config_clustering_tuning.json"
)

# Output directory for all summary files
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sum_eval_metrics_summary_clustering_tuning"
)

def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Initialize the analyzer
    analyzer = Analyzer()

    # 1. Generate Odometry Error Comparison Table
    odom_save_path = os.path.join(OUTPUT_DIR, "odom_error_comparison_summary.csv")
    print(f"Generating Odometry Error Comparison Table to {odom_save_path}...")
    try:
        analyzer.generate_odom_error_comparison_table(
            config=config,
            percentile=0.90,
            save_path=odom_save_path
        )
    except Exception as e:
        print(f"Failed to generate odom error table: {e}")

    # 2. Generate PC Quality Comparison Table
    pc_save_path = os.path.join(OUTPUT_DIR, "pc_quality_comparison_summary.csv")
    print(f"Generating PC Quality Comparison Table to {pc_save_path}...")
    try:
        analyzer.generate_pc_quality_comparison_table(
            config=config,
            percentile=0.90,
            save_path=pc_save_path
        )
    except Exception as e:
        print(f"Failed to generate pc quality table: {e}")

    # 3. Generate Dataset Qualities Table
    print("Generating Dataset Qualities Table...")
    dataset_qualities = []
    
    for dataset_name, methods in config.items():
        if not methods:
            continue
            
        # We only need one method's path to extract dataset-wide metrics (distance, frames)
        # as they should be identical across methods for the same dataset recordings.
        first_method_path = list(methods.values())[0]
        
        try:
            summary_stats = analyzer.get_summary_statistics_from_csvs(first_method_path)
            
            num_trials = len(summary_stats["trial_distances"])
            total_distance = summary_stats["total_distance"]
            avg_trial_distance = np.mean(summary_stats["trial_distances"]) if num_trials > 0 else 0
            total_frames = summary_stats["num_frames"]
            avg_frames_per_trial = np.mean(summary_stats["trial_frames"]) if num_trials > 0 else 0
            
            dataset_qualities.append({
                "Dataset": dataset_name,
                "Num Trials": num_trials,
                "Total Distance (m)": total_distance,
                "Average Trial Distance (m)": avg_trial_distance,
                "Total Frames": total_frames,
                "Average Frames per Trial": avg_frames_per_trial
            })
        except Exception as e:
            print(f"Failed to extract qualities for dataset '{dataset_name}': {e}")

    if dataset_qualities:
        dq_df = pd.DataFrame(dataset_qualities)
        dq_save_path = os.path.join(OUTPUT_DIR, "dataset_qualities.csv")
        dq_df.to_csv(dq_save_path, index=False)
        print(f"Saved dataset qualities to {dq_save_path}")
        print(dq_df.to_string(index=False))

if __name__ == "__main__":
    main()
