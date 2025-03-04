#!/usr/bin/env python
"""
Utility script to sync offline wandb runs
"""

import argparse
import subprocess
import glob

def main():
    parser = argparse.ArgumentParser(description="Sync offline wandb runs")
    parser.add_argument("--all", action="store_true", help="Sync all unsynced wandb runs")
    parser.add_argument("--run-id", type=str, help="Specific run ID to sync (e.g., '20250303_184216')")
    
    args = parser.parse_args()
    
    if not args.all and not args.run_id:
        parser.error("Please specify either --all or --run-id")
    
    # Find wandb directories to sync
    if args.all:
        # Sync all unsynced runs
        print("Syncing all unsynced wandb runs...")
        subprocess.run(["wandb", "sync", "--include-globs", "wandb/run-*"])
    else:
        # Sync specific run
        run_pattern = f"wandb/run-*{args.run_id}*"
        matching_runs = glob.glob(run_pattern)
        
        if not matching_runs:
            print(f"No wandb runs found matching pattern: {run_pattern}")
            return
        
        for run_dir in matching_runs:
            print(f"Syncing wandb run: {run_dir}")
            subprocess.run(["wandb", "sync", run_dir])
    
    print("Sync complete!")

if __name__ == "__main__":
    main() 