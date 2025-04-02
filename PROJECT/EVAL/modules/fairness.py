import numpy as np
import pandas as pd
from .model_builder import build_predictor

def build_balanced_predictor(diagnoses_df, demographics_df, target_icd_code, balance_by='race'):
    """
    Build a model with balanced demographic representation.
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        demographics_df: DataFrame with demographic data
        target_icd_code: The ICD code to predict
        balance_by: Demographic variable to balance by
        
    Returns:
        G: Balanced patient network graph
        target_patients: Patients with target diagnosis
        non_target_patients: Patients without target diagnosis
    """
    # Identify the column to balance by
    if balance_by not in demographics_df.columns:
        print(f"Cannot balance by {balance_by}, column not found")
        return None
    
    # Get demographic distribution
    demo_counts = demographics_df[balance_by].value_counts()
    min_group_size = max(100, int(demo_counts.min() * 1.5))  # Ensure minimum viable size
    
    # Sample approximately equal numbers from each demographic group
    sampled_patients = []
    
    for group in demo_counts.index:
        # Get patients in this group
        group_patients = demographics_df[demographics_df[balance_by] == group]['subject_id'].tolist()
        
        # Split into those with and without target diagnosis
        patients_with_target = list(set(group_patients).intersection(
            set(diagnoses_df[diagnoses_df['icd_code'] == target_icd_code]['subject_id'])))
        patients_without_target = list(set(group_patients) - set(patients_with_target))
        
        # Calculate sampling proportions
        target_ratio = len(patients_with_target) / len(group_patients) if group_patients else 0
        
        # Sample sizes, preserving the target/non-target ratio within each group
        n_target = min(int(min_group_size * target_ratio), len(patients_with_target))
        n_non_target = min(min_group_size - n_target, len(patients_without_target))
        
        # Sample
        if patients_with_target and n_target > 0:
            sampled_target = np.random.choice(patients_with_target, size=n_target, replace=False)
            sampled_patients.extend(sampled_target)
        
        if patients_without_target and n_non_target > 0:
            sampled_non_target = np.random.choice(patients_without_target, size=n_non_target, replace=False)
            sampled_patients.extend(sampled_non_target)
    
    # Filter diagnoses to only these patients
    balanced_diagnoses = diagnoses_df[diagnoses_df['subject_id'].isin(sampled_patients)]
    
    print(f"Created balanced sample with {len(sampled_patients)} patients")
    # Check demographic balance in the sample
    demo_balance = demographics_df[demographics_df['subject_id'].isin(sampled_patients)][balance_by].value_counts()
    print(f"\nBalanced sample {balance_by} distribution:")
    print(demo_balance)
    
    # Now build predictor as usual
    return build_predictor(
        balanced_diagnoses,
        demographics_df,
        target_icd_code,
        include_demographics=True
    )

def calculate_disparity_metrics(results_dict):
    """
    Calculate disparity metrics from fairness evaluation results.
    
    Args:
        results_dict: Dictionary of results by demographic group
        
    Returns:
        Dictionary with disparity metrics
    """
    disparity_metrics = {}
    
    for demo_var, group_results in results_dict.items():
        # Convert to DataFrame for analysis
        comparison_df = pd.DataFrame(group_results).T
        
        if len(comparison_df) > 1:
            # Calculate disparities
            max_fnr = comparison_df['false_neg_rate'].max()
            min_fnr = comparison_df['false_neg_rate'].min()
            fnr_disparity = max_fnr / min_fnr if min_fnr > 0 else float('inf')
            
            max_sens = comparison_df['sensitivity'].max()
            min_sens = comparison_df['sensitivity'].min()
            sens_disparity = max_sens / min_sens if min_sens > 0 else float('inf')
            
            # Calculate additional fairness metrics
            # Equal opportunity difference (difference in TPR)
            eod = max_sens - min_sens
            
            # Disparate impact (ratio of favorable outcomes)
            di = min_sens / max_sens if max_sens > 0 else 0
            
            # Most affected groups
            most_affected_fnr = comparison_df['false_neg_rate'].idxmax()
            most_affected_sens = comparison_df['sensitivity'].idxmin()
            
            disparity_metrics[demo_var] = {
                'fnr_disparity': fnr_disparity,
                'sensitivity_disparity': sens_disparity,
                'equal_opportunity_diff': eod,
                'disparate_impact': di,
                'most_affected_fnr': most_affected_fnr,
                'most_affected_sens': most_affected_sens,
                'significant_disparity': fnr_disparity > 1.2 or sens_disparity > 1.2
            }
    
    return disparity_metrics