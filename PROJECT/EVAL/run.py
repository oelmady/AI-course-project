import os

# Create Results directory if it doesn't exist
if not os.path.exists('Results'):
    os.makedirs('Results')

# Import modules
from modules.data_loader import load_diagnoses_data, load_demographics_data
from modules.demographic_analyzer import analyze_demographic_risk_factors
from modules.evaluation import (
    evaluate_threshold_curve_with_demographics,
    compare_models_with_without_demographics
)

def main():
    """Main execution function for KNN diagnosis evaluation with demographics"""
    # Load data
    diagnoses_df = load_diagnoses_data()
    demographics_df = load_demographics_data()
    
    # Mental health diagnosis
    target_icd = "F30"

    compare_models_with_without_demographics(
        diagnoses_df,
        demographics_df,
        target_icd,
        k=11,
        n_folds=5,
        similarity_threshold=0.15,
        sample_size=5000,
        parallel_jobs=-1
    )
    
    evaluate_threshold_curve_with_demographics(
        diagnoses_df,
        demographics_df,
        target_icd,
        k=11,
        sample_size=5000
    )

if __name__ == "__main__":
    main()