"""
Author: Metin Zontul
Date: October 2026
Description: Re-calculates Table 1 metrics using the exact 15-run evaluations from 
             statistical_15_runs_data.csv. Applies sample standard deviation (ddof=1) 
             across all seeds and the overall 15-run pool to resolve Reviewer Point 1.
"""

import pandas as pd
import numpy as np

def generate_corrected_table1():
    # Load the verified raw 15-run evaluation data
    df = pd.read_csv("statistical_15_runs_data.csv")
    
    # Target accuracy columns based on the CSV structure
    s42_cols = [f"S42_F{i}_Acc" for i in range(1, 6)]
    s1024_cols = [f"S1024_F{i}_Acc" for i in range(1, 6)]
    s2026_cols = [f"S2026_F{i}_Acc" for i in range(1, 6)]
    all_cols = s42_cols + s1024_cols + s2026_cols
    
    results = []
    
    for index, row in df.iterrows():
        model = row['Model']
        
        # Calculate Mean and SD (ddof=1) for Seed 42
        s42_vals = row[s42_cols].astype(float)
        s42_str = f"{s42_vals.mean():.2f} \\pm {s42_vals.std(ddof=1):.2f}"
        
        # Calculate Mean and SD (ddof=1) for Seed 1024
        s1024_vals = row[s1024_cols].astype(float)
        s1024_str = f"{s1024_vals.mean():.2f} \\pm {s1024_vals.std(ddof=1):.2f}"
        
        # Calculate Mean and SD (ddof=1) for Seed 2026
        s2026_vals = row[s2026_cols].astype(float)
        s2026_str = f"{s2026_vals.mean():.2f} \\pm {s2026_vals.std(ddof=1):.2f}"
        
        # Calculate Overall Mean and SD (ddof=1) across all 15 evaluations directly
        all_vals = row[all_cols].astype(float)
        overall_str = f"{all_vals.mean():.2f} \\pm {all_vals.std(ddof=1):.2f}"
        
        results.append({
            "Architecture": model,
            "Seed 42 (Mean $\\pm$ SD, \\%)": s42_str,
            "Seed 1024 (Mean $\\pm$ SD, \\%)": s1024_str,
            "Seed 2026 (Mean $\\pm$ SD, \\%)": s2026_str,
            "Overall 15-Run (Mean $\\pm$ SD, \\%)": overall_str
        })
        
    final_df = pd.DataFrame(results)
    
    print("--- Updated Table 1 Preview ---")
    print(final_df.to_string(index=False))
    
    print("\n--- LaTeX Code for Overleaf ---")
    print(final_df.to_latex(index=False, escape=False))
    
    final_df.to_csv("Corrected_Table_1.csv", index=False)
    print("\n[!] Düzeltilmiş tablo 'Corrected_Table_1.csv' olarak kaydedildi.")

if __name__ == "__main__":
    generate_corrected_table1()