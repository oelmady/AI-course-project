import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def analyze_demographic_error_patterns(all_predictions, all_true_labels, patient_ids, demographics_df):
    """
    Analyzes how prediction errors vary across different demographic groups.
    
    Args:
        all_predictions: Array of predicted probabilities
        all_true_labels: Array of true labels
        patient_ids: List of patient IDs corresponding to predictions
        demographics_df: DataFrame with demographic data
        
    Returns:
        DataFrame with error analysis by demographic group
    """
    # Create DataFrame with predictions and true labels
    results_df = pd.DataFrame({
        'subject_id': patient_ids,
        'true_label': all_true_labels,
        'prediction': all_predictions,
        'error': [1 if abs(pred - label) > 0.5 else 0 for pred, label in zip(all_predictions, all_true_labels)]
    })
    
    # Merge with demographics
    analysis_df = results_df.merge(demographics_df, on='subject_id', how='inner')
    
    print(f"Analyzing errors across {len(analysis_df)} patients with demographic data")
    
    # Calculate error rates by demographic groups
    demographic_factors = ['gender', 'insurance', 'race', 'language', 'marital_status']
    demographic_factors = [col for col in demographic_factors if col in demographics_df.columns]
    
    error_analysis = {}
    
    for factor in demographic_factors:
        # Group error analysis
        group_error = analysis_df.groupby(factor)['error'].mean().reset_index()
        group_error.columns = [factor, 'error_rate']
        
        # Group count
        group_count = analysis_df.groupby(factor)['subject_id'].count().reset_index()
        group_count.columns = [factor, 'count']
        
        # False negative analysis (for patients with true_label=1)
        positives_df = analysis_df[analysis_df['true_label'] == 1]
        if len(positives_df) > 0:
            false_neg = positives_df.groupby(factor).apply(
                lambda x: sum(x['prediction'] < 0.5) / len(x) if len(x) > 0 else 0
            ).reset_index()
            false_neg.columns = [factor, 'false_negative_rate']
        else:
            false_neg = pd.DataFrame({factor: group_error[factor], 'false_negative_rate': 0})
            
        # False positive analysis (for patients with true_label=0)
        negatives_df = analysis_df[analysis_df['true_label'] == 0]
        if len(negatives_df) > 0:
            false_pos = negatives_df.groupby(factor).apply(
                lambda x: sum(x['prediction'] >= 0.5) / len(x) if len(x) > 0 else 0
            ).reset_index()
            false_pos.columns = [factor, 'false_positive_rate']
        else:
            false_pos = pd.DataFrame({factor: group_error[factor], 'false_positive_rate': 0})
        
        # Merge all metrics
        group_stats = group_error.merge(group_count, on=factor)
        group_stats = group_stats.merge(false_neg, on=factor)
        group_stats = group_stats.merge(false_pos, on=factor)
        
        # Store in dictionary
        error_analysis[factor] = group_stats
    
    # Print summary
    print("\n=== Error Analysis by Demographic Groups ===")
    for factor, stats in error_analysis.items():
        print(f"\n{factor.upper()}")
        for _, row in stats.sort_values('error_rate', ascending=False).iterrows():
            print(f"  {row[factor]}: Error Rate={row['error_rate']:.3f}, "
                 f"FNR={row['false_negative_rate']:.3f}, "
                 f"FPR={row['false_positive_rate']:.3f}, "
                 f"Count={row['count']}")
    
    return error_analysis

def analyze_demographic_risk_factors(diagnoses_df, demographics_df, target_icd_code):
    """
    Identifies which demographic factors are most strongly associated with the target diagnosis.
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        demographics_df: DataFrame with demographic data
        target_icd_code: The ICD code being analyzed
        
    Returns:
        DataFrame: Statistical association between demographics and target diagnosis
    """
    # Get patients with the target diagnosis
    target_patients = diagnoses_df[diagnoses_df['icd_code'] == target_icd_code]['subject_id'].unique()
    target_patients_set = set(target_patients)
    
    # Merge demographics with target diagnosis indicator
    demographics_with_target = demographics_df.copy()
    demographics_with_target['has_target'] = demographics_with_target['subject_id'].isin(target_patients_set).astype(int)
    
    # Count how many patients have the target diagnosis
    target_count = demographics_with_target['has_target'].sum()
    total_count = len(demographics_with_target)
    overall_rate = target_count / total_count
    
    print(f"Overall prevalence: {overall_rate:.4f} ({target_count}/{total_count})")
    
    # Analyze each demographic factor
    demographic_factors = ['gender', 'insurance', 'race', 'language', 'marital_status']
    demographic_factors = [col for col in demographic_factors if col in demographics_with_target.columns]
    
    result_tables = []
    
    for factor in demographic_factors:
        # Create contingency table
        contingency = pd.crosstab(
            demographics_with_target[factor],
            demographics_with_target['has_target'],
            normalize='columns'
        )
        
        # Add counts
        counts = pd.crosstab(
            demographics_with_target[factor],
            demographics_with_target['has_target']
        )
        contingency['count_0'] = counts[0]
        contingency['count_1'] = counts[1]
        contingency['total_count'] = contingency['count_0'] + contingency['count_1']
        
        # Calculate prevalence
        group_prevalence = demographics_with_target.groupby(factor)['has_target'].mean()
        contingency['prevalence'] = group_prevalence
        
        # Calculate prevalence ratio compared to overall prevalence
        contingency['prevalence_ratio'] = contingency['prevalence'] / overall_rate
        
        # Add p-values from chi-square test
        from scipy.stats import chi2_contingency
        p_values = {}
        
        for group in contingency.index:
            # Create 2x2 contingency table for this group vs all others
            group_df = demographics_with_target[demographics_with_target[factor] == group]
            others_df = demographics_with_target[demographics_with_target[factor] != group]
            
            table = np.array([
                [sum(group_df['has_target']), len(group_df) - sum(group_df['has_target'])],
                [sum(others_df['has_target']), len(others_df) - sum(others_df['has_target'])]
            ])
            
            # Run chi-square test
            try:
                chi2, p, _, _ = chi2_contingency(table)
                p_values[group] = p
            except:
                p_values[group] = 1.0
        
        contingency['p_value'] = pd.Series(p_values)
        
        # Reset index and add factor name
        contingency = contingency.reset_index()
        contingency['factor'] = factor
        
        result_tables.append(contingency)
    
    # Combine all results
    all_results = pd.concat(result_tables, ignore_index=True)
    
    # Display results
    print(f"\nDemographic Risk Factor Analysis for {target_icd_code}:")
    for factor in demographic_factors:
        factor_results = all_results[all_results['factor'] == factor]
        print(f"\n{factor.upper()} as a risk factor:")
        for _, row in factor_results.sort_values('prevalence_ratio', ascending=False).iterrows():
            significance = "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"  {row[factor]}: Prevalence={row['prevalence']*100:.2f}%, "
                 f"Ratio={row['prevalence_ratio']:.2f}x{significance}, "
                 f"Count={row['total_count']} (with condition: {row['count_1']})")
    
    return all_results

def calculate_demographic_metrics(predictions_df, threshold, demographic_columns):
    """
    Calculate metrics for demographic groups at a specific threshold.
    
    Args:
        predictions_df: DataFrame with predictions and demographics
        threshold: Decision threshold for binary classification
        demographic_columns: List of demographic columns to analyze
        
    Returns:
        Dictionary with metrics for each demographic group
    """
    # Create binary predictions
    predictions_df['binary_pred'] = predictions_df['prediction'].apply(lambda p: 1 if p >= threshold else 0)
    
    # Overall metrics
    tn, fp, fn, tp = confusion_matrix(
        predictions_df['true_label'], 
        predictions_df['binary_pred']
    ).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    result = {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'false_negatives': fn,
        'false_positives': fp,
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    }
    
    # Calculate metrics for each demographic group
    for demo_col in demographic_columns:
        if demo_col in predictions_df.columns:
            for group_value in predictions_df[demo_col].unique():
                group_df = predictions_df[predictions_df[demo_col] == group_value]
                # Only include groups with sufficient patients
                if len(group_df) >= 20:  
                    try:
                        group_cm = confusion_matrix(
                            group_df['true_label'], 
                            group_df['binary_pred'],
                            labels=[0, 1]
                        ).ravel()
                        
                        if len(group_cm) == 4:  # Ensure we have all 4 values (tn, fp, fn, tp)
                            group_tn, group_fp, group_fn, group_tp = group_cm
                            
                            group_sensitivity = group_tp / (group_tp + group_fn) if (group_tp + group_fn) > 0 else 0
                            group_specificity = group_tn / (group_tn + group_fp) if (group_tn + group_fp) > 0 else 0
                            group_fnr = group_fn / (group_fn + group_tp) if (group_fn + group_tp) > 0 else 0
                            
                            # Add to results
                            result[f'{demo_col}_{group_value}_fnr'] = group_fnr
                            result[f'{demo_col}_{group_value}_sensitivity'] = group_sensitivity
                            result[f'{demo_col}_{group_value}_patients'] = len(group_df)
                    except Exception as e:
                        print(f"Error calculating metrics for {demo_col}={group_value}: {e}")
                        
    return result