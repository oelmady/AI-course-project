"""
KNN Fairness Analysis Module

This module provides functionality for assessing fairness in KNN-based 
medical diagnosis prediction models across demographic groups.
"""

from modules.data_loader import load_diagnoses_data, load_demographics_data
from modules.fairness import build_balanced_predictor
from modules.evaluation import evaluate_demographic_fairness

def main():
    """Main function to demonstrate fairness evaluation functionality"""
    # Load data
    diagnoses_df = load_diagnoses_data()
    demographics_df = load_demographics_data()
    
    # Example target diagnosis code (hypertension)
    target_icd = "I10"
    
    # Option 1: Build a model with balanced demographic representation
    print("\n=== Building Balanced Model ===")
    G, target_patients, non_target_patients = build_balanced_predictor(
        diagnoses_df, 
        demographics_df, 
        target_icd, 
        balance_by='race'
    )
    print(f"Built balanced model with {len(target_patients)} positive and {len(non_target_patients)} negative examples")
    
    # Option 2: Evaluate fairness across demographic groups
    print("\n=== Evaluating Fairness Across Demographic Groups ===")
    _, disparity_metrics = evaluate_demographic_fairness(
        diagnoses_df,
        demographics_df,
        target_icd,
        sample_size=3000
    )
    
    # Summarize findings
    print("\n=== Fairness Evaluation Summary ===")
    has_disparity = any(metrics['significant_disparity'] for metrics in disparity_metrics.values())
    
    if has_disparity:
        print("Significant disparities detected in model performance across demographic groups.")
        
        # Display the specific disparities
        for demo_var, metrics in disparity_metrics.items():
            if metrics['significant_disparity']:
                print(f"\n  For {demo_var}:")
                print(f"    - FNR disparity: {metrics['fnr_disparity']:.2f}x (most affected: {metrics['most_affected_fnr']})")
                print(f"    - Sensitivity disparity: {metrics['sensitivity_disparity']:.2f}x (most affected: {metrics['most_affected_sens']})")
        
        print("\nPossible mitigations:")
        print("1. Use demographic-balanced training data")
        print("2. Apply different decision thresholds for different groups")
        print("3. Incorporate demographic features with higher weight")
    else:
        print("✓ No significant disparities detected across demographic groups")
    
    print("\nResults saved to Results/ directory")

if __name__ == "__main__":
    main()