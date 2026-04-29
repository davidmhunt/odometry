import sys
import os
import json
import pandas as pd
import numpy as np
import ast

# Add the repository root to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from odometry.analyzers.analyzer import Analyzer

# Fold configurations to process - Update this list with your actual fold config filenames
FOLDS = [
    "analyzer_config_clustering_tuning.json",
    "analyzer_config_clustering_tuning_f2.json"
]

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer_configs")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_fold_summary")

def main():
    """Processes multiple evaluation folds and generates aggregated summary statistics.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    analyzer = Analyzer()
    
    odom_dfs = []
    pc_dfs = []
    quality_records = []
    
    print(f"Starting Multi-Fold Aggregation...")
    print(f"Config Directory: {CONFIG_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    for fold_file in FOLDS:
        config_path = os.path.join(CONFIG_DIR, fold_file)
        if not os.path.exists(config_path):
            print(f"Skipping {fold_file}: File not found at {config_path}")
            continue
            
        print(f"\n{'='*40}")
        print(f"Processing Fold: {fold_file}")
        print(f"{'='*40}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
            
        fold_label = fold_file.replace(".json", "")
        
        # 1. Odometry Error Comparison
        odom_save_path = os.path.join(OUTPUT_DIR, f"odom_error_{fold_label}.csv")
        print(f"Generating Odometry Error Table for {fold_label}...")
        try:
            df_odom = analyzer.generate_odom_error_comparison_table(
                config=config, 
                percentile=0.90, 
                save_path=odom_save_path
            )
            if df_odom is not None:
                odom_dfs.append(df_odom)
        except Exception as e:
            print(f"Error generating odom table for {fold_label}: {e}")
            
        # 2. PC Quality Comparison
        pc_save_path = os.path.join(OUTPUT_DIR, f"pc_quality_{fold_label}.csv")
        print(f"Generating PC Quality Table for {fold_label}...")
        try:
            df_pc = analyzer.generate_pc_quality_comparison_table(
                config=config, 
                percentile=0.90, 
                save_path=pc_save_path
            )
            if df_pc is not None:
                pc_dfs.append(df_pc)
        except Exception as e:
            print(f"Error generating PC quality table for {fold_label}: {e}")
            
        # 3. Dataset Qualities
        print(f"Extracting Dataset Qualities for {fold_label}...")
        for dataset_name, methods in config.items():
            if not methods:
                continue
                
            # We use the first method to get dataset-wide stats (distance, frames)
            first_method_path = list(methods.values())[0]
            try:
                stats = analyzer.get_summary_statistics_from_csvs(first_method_path)
                
                num_trials = len(stats["trial_distances"])
                total_distance = stats["total_distance"]
                avg_trial_distance = np.mean(stats["trial_distances"]) if num_trials > 0 else 0
                total_frames = stats["num_frames"]
                avg_frames_per_trial = np.mean(stats["trial_frames"]) if num_trials > 0 else 0
                
                quality_records.append({
                    "Dataset": dataset_name,
                    "Fold": fold_label,
                    "Num Trials": num_trials,
                    "Total Distance (m)": total_distance,
                    "Average Trial Distance (m)": avg_trial_distance,
                    "Total Frames": total_frames,
                    "Average Frames per Trial": avg_frames_per_trial
                })
            except Exception as e:
                print(f"Failed to extract qualities for dataset '{dataset_name}' in {fold_label}: {e}")

    # --- Cross-Fold Aggregation ---

    if odom_dfs:
        print(f"\n{'-'*40}")
        print("Aggregating Odometry Errors across folds...")
        combined_odom = pd.concat(odom_dfs)
        
        # Compute Mean and Std across the fold scores
        final_odom_mean = combined_odom.groupby(level=[0, 1]).mean()
        final_odom_std = combined_odom.groupby(level=[0, 1]).std()
        
        # Combine and reorder levels to (Metric, Statistic, AggregateType)
        final_odom = pd.concat([final_odom_mean, final_odom_std], axis=1, keys=['Mean', 'Std'])
        final_odom = final_odom.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)
        
        summary_save_path = os.path.join(OUTPUT_DIR, "odom_error_multi_fold_summary.csv")
        final_odom.to_csv(summary_save_path)
        print(f"Saved Multi-Fold Odom Summary to {summary_save_path}")
        print(final_odom.to_string())

    if pc_dfs:
        print(f"\n{'-'*40}")
        print("Aggregating PC Qualities across folds...")
        combined_pc = pd.concat(pc_dfs)
        
        final_pc_mean = combined_pc.groupby(level=[0, 1]).mean()
        final_pc_std = combined_pc.groupby(level=[0, 1]).std()
        
        final_pc = pd.concat([final_pc_mean, final_pc_std], axis=1, keys=['Mean', 'Std'])
        final_pc = final_pc.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)
        
        summary_save_path = os.path.join(OUTPUT_DIR, "pc_quality_multi_fold_summary.csv")
        final_pc.to_csv(summary_save_path)
        print(f"Saved Multi-Fold PC Quality Summary to {summary_save_path}")
        print(final_pc.to_string())

    if quality_records:
        print(f"\n{'-'*40}")
        print("Aggregating Dataset Qualities across folds...")
        df_q = pd.DataFrame(quality_records)
        
        # Aggregate: Sum for totals, Mean for averages
        agg_q = df_q.groupby("Dataset").agg({
            "Num Trials": "sum",
            "Total Distance (m)": "sum",
            "Average Trial Distance (m)": "mean",
            "Total Frames": "sum",
            "Average Frames per Trial": "mean"
        })
        
        summary_save_path = os.path.join(OUTPUT_DIR, "dataset_qualities_multi_fold.csv")
        agg_q.to_csv(summary_save_path)
        print(f"Saved Multi-Fold Dataset Qualities to {summary_save_path}")
        print(agg_q.to_string())

if __name__ == "__main__":
    main()
