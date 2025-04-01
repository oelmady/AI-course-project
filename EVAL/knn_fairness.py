from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from knn_full_eval import build_predictor, evaluate_threshold_curve_with_demographics


def build_balanced_predictor(diagnoses_df, demographics_df, target_icd_code, balance_by='race'):
    """
    Build a more balanced model by ensuring demographic representation.
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
    
def measure_demographic_fairness(diagnoses_df, demographics_df, target_icd_code, sample_size=3000):
    """
    Measures fairness metrics across demographic groups for a given diagnosis.
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        demographics_df: DataFrame with demographic data
        target_icd_code: The ICD code to analyze
        sample_size: Size of sample to use for analysis
    """
    print(f"\n=== Measuring Prediction Fairness for {target_icd_code} ===")
    
    # Get demographic distributions
    demo_counts = {}
    for col in ['gender', 'race', 'insurance', 'language']:
        if col in demographics_df.columns:
            counts = demographics_df[col].value_counts()
            total = counts.sum()
            percentages = (counts / total * 100).round(1)
            
            # Format as "count (percentage%)"
            formatted = [f"{count} ({pct}%)" for count, pct in zip(counts, percentages)]
            demo_counts[col] = dict(zip(counts.index, formatted))
    
    print("\nDemographic distribution in dataset:")
    for col, counts in demo_counts.items():
        print(f"\n{col.upper()}:")
        for group, count in counts.items():
            print(f"  {group}: {count}")
    
    # Build separate models for each major demographic group
    # This allows us to compare performance across groups
    results = {}
    
    # Analysis for major demographic variables
    for demo_var in ['race', 'gender', 'insurance']:
        if demo_var not in demographics_df.columns:
            continue
            
        # Get the largest groups that collectively cover at least 90% of patients
        counts = demographics_df[demo_var].value_counts()
        total = counts.sum()
        cumulative_pct = 0
        groups_to_analyze = []
        
        for group, count in counts.items():
            cumulative_pct += count/total * 100
            groups_to_analyze.append(group)
            if cumulative_pct >= 90:
                break
        
        print(f"\nAnalyzing {demo_var} groups: {', '.join(groups_to_analyze)}")
        
        # For each group, evaluate model performance
        group_results = {}
        for group in groups_to_analyze:
            print(f"\nEvaluating model for {demo_var}={group}")
            
            # Get patients in this demographic group
            group_patients = demographics_df[demographics_df[demo_var] == group]['subject_id']
            
            # Filter diagnoses to only these patients
            group_diagnoses = diagnoses_df[diagnoses_df['subject_id'].isin(group_patients)]
            
            # Skip if too few patients
            if len(group_diagnoses['subject_id'].unique()) < 100:
                print(f"  Too few patients with {demo_var}={group}, skipping")
                continue
                
            # Run stratified evaluation
            threshold_results = evaluate_threshold_curve_with_demographics(
                group_diagnoses,
                demographics_df,
                target_icd_code,
                k=10,
                sample_size=min(sample_size, len(group_diagnoses['subject_id'].unique())),
                demographic_group=(demo_var, group),
                include_demographics=False  # We're already filtering by demographic
            )
            
            # Store results for comparison
            optimal_threshold = threshold_results.loc[threshold_results['f1'].idxmax()]
            group_results[group] = {
                'sensitivity': optimal_threshold['sensitivity'],
                'specificity': optimal_threshold['specificity'],
                'precision': optimal_threshold['precision'],
                'f1': optimal_threshold['f1'],
                'false_neg_rate': optimal_threshold['false_negatives'] / 
                                (optimal_threshold['false_negatives'] + 
                                 (optimal_threshold['precision'] * optimal_threshold['false_positives'] / 
                                  (1 - optimal_threshold['precision']) if optimal_threshold['precision'] < 1 else 0)),
                'optimal_threshold': optimal_threshold['threshold']
            }
            
        # Save results for this demographic variable
        results[demo_var] = group_results
    
    # Compare results across demographic groups
    print("\n=== Fairness Comparison Across Demographic Groups ===")
    
    for demo_var, group_results in results.items():
        print(f"\n{demo_var.upper()} COMPARISON:")
        
        # Create a DataFrame for easier comparison
        comparison_df = pd.DataFrame(group_results).T
        
        # Format for display
        with pd.option_context('display.precision', 3):
            print(comparison_df)
            
        # Calculate disparities
        if len(comparison_df) > 1:
            max_fnr = comparison_df['false_neg_rate'].max()
            min_fnr = comparison_df['false_neg_rate'].min()
            fnr_disparity = max_fnr / min_fnr if min_fnr > 0 else float('inf')
            
            max_sens = comparison_df['sensitivity'].max()
            min_sens = comparison_df['sensitivity'].min()
            sens_disparity = max_sens / min_sens if min_sens > 0 else float('inf')
            
            print(f"\nDisparities:")
            print(f"  False Negative Rate Disparity: {fnr_disparity:.2f}x")
            print(f"  Sensitivity Disparity: {sens_disparity:.2f}x")
            
            # Flag significant disparities
            if fnr_disparity > 1.2:
                print(f"  ⚠️ Warning: Significant disparity in false negative rates across {demo_var} groups")
                most_affected = comparison_df['false_neg_rate'].idxmax()
                print(f"     Group most affected: {most_affected}")
    
    # Visualize fairness metrics
    plot_fairness_comparison(results, target_icd_code)
    
    return results

def plot_fairness_comparison(results, target_icd_code):
    """
    Visualizes fairness metrics across demographic groups.
    
    Args:
        results: Dictionary with results from measure_demographic_fairness
        target_icd_code: The ICD code being analyzed
    """
    n_vars = len(results)
    if n_vars == 0:
        return
        
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 6*n_vars), constrained_layout=True)
    
    if n_vars == 1:
        axes = [axes]
        
    metrics = ['sensitivity', 'specificity', 'precision', 'false_neg_rate']
    colors = ['blue', 'green', 'purple', 'red']
    
    for i, (demo_var, group_results) in enumerate(results.items()):
        # Convert to DataFrame for plotting
        df = pd.DataFrame(group_results).T
        
        # Set up bar positions
        groups = df.index
        x = np.arange(len(groups))
        width = 0.2
        
        # Plot each metric as a group of bars
        for j, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (j - 1.5)
            axes[i].bar(x + offset, df[metric], width, label=metric.replace('_', ' ').title(), color=color, alpha=0.7)
        
        # Add reference line for false negative rate
        if 'false_neg_rate' in df.columns:
            axes[i].axhline(y=df['false_neg_rate'].min(), color='red', linestyle='--', alpha=0.5)
        
        # Set up labels and legend
        axes[i].set_ylabel('Score')
        axes[i].set_title(f'Fairness Metrics by {demo_var.title()} for {target_icd_code}')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(groups)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Ensure y-axis starts at 0
        axes[i].set_ylim(0, min(1.0, df[metrics].max().max() * 1.1))
    
    plt.savefig(f'fairness_comparison_{target_icd_code}.png')
    plt.show()