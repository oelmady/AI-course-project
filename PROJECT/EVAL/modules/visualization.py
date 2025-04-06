import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

def plot_evaluation_metrics(true_labels, predictions, binary_predictions, target_icd_code):
    """
    Plot ROC curve and confusion matrix for model evaluation.
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted probabilities
        binary_predictions: Array of binary predictions
        target_icd_code: ICD code being predicted
    """
    from sklearn.metrics import roc_auc_score
    
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(true_labels, predictions):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {target_icd_code}')
    plt.legend()
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, binary_predictions)
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f'Results/evaluation_{target_icd_code}_with_demographics.png')
    plt.close()

def plot_demographic_error_analysis(error_analysis):
    """
    Creates visualizations of error rates across demographic groups.
    
    Args:
        error_analysis: Dictionary with error analysis by demographic factor
    """
    n_factors = len(error_analysis)
    if n_factors == 0:
        print("No demographic factors to plot")
        return
        
    fig, axes = plt.subplots(n_factors, 1, figsize=(14, 5*n_factors), constrained_layout=True)
    
    if n_factors == 1:
        axes = [axes]  # Make iterable for consistent access
    
    for i, (factor, data) in enumerate(error_analysis.items()):
        # Sort by error rate for better visualization
        data = data.sort_values('error_rate', ascending=False)
        
        # Create a bar width that reflects the group size
        sizes = data['count'] / data['count'].max() * 0.8 + 0.2  # Scale between 0.2 and 1.0
        x_pos = np.arange(len(data))
        
        # Plot overall error rate
        bars1 = axes[i].bar(x_pos - 0.2, data['error_rate'], width=0.2, 
                          alpha=0.7, label='Overall Error Rate')
        
        # Plot false negative rate
        bars2 = axes[i].bar(x_pos, data['false_negative_rate'], width=0.2, 
                          alpha=0.7, label='False Negative Rate')
        
        # Plot false positive rate
        bars3 = axes[i].bar(x_pos + 0.2, data['false_positive_rate'], width=0.2, 
                          alpha=0.7, label='False Positive Rate')
        
        # Add count labels on top of bars
        for j, count in enumerate(data['count']):
            axes[i].text(x_pos[j], max(data['error_rate'].iloc[j], 
                                    data['false_negative_rate'].iloc[j],
                                    data['false_positive_rate'].iloc[j]) + 0.02, 
                       f"n={count}", ha='center', va='bottom', fontsize=9)
        
        # Set x-axis labels and title
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(data[factor])
        axes[i].set_title(f'Error Analysis by {factor.title()}')
        axes[i].set_ylabel('Rate')
        axes[i].set_ylim(0, min(1.0, data[['error_rate', 'false_negative_rate', 'false_positive_rate']].max().max() * 1.2))
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Rotate x-labels if needed
        if len(data) > 5 or max(len(str(x)) for x in data[factor]) > 10:
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.savefig('Results/demographic_error_analysis.png')
    plt.close()

def plot_demographic_risk_factors(all_results, target_icd_code):
    """
    Plot the risk factors by demographic groups.
    """
    factors = all_results['factor'].unique()
    n_factors = len(factors)
    
    fig, axes = plt.subplots(n_factors, 1, figsize=(14, 6*n_factors), constrained_layout=True)
    
    if n_factors == 1:
        axes = [axes]
    
    for i, factor in enumerate(factors):
        factor_data = all_results[all_results['factor'] == factor].sort_values('prevalence_ratio', ascending=False)
        
        # Plot bars
        bars = axes[i].bar(factor_data[factor], factor_data['prevalence_ratio'], alpha=0.7)
        
        # Add counts
        for j, (_, row) in enumerate(factor_data.iterrows()):
            axes[i].text(j, row['prevalence_ratio'] + 0.05, 
                       f"n={row['total_count']}\n({row['count_1']} cases)", 
                       ha='center', va='bottom', fontsize=9)
        
        # Add reference line for ratio=1.0
        axes[i].axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        
        # Labels and title
        axes[i].set_title(f'Prevalence Ratio by {factor.title()} for {target_icd_code}')
        axes[i].set_ylabel('Prevalence Ratio')
        axes[i].set_ylim(0, max(3.0, factor_data['prevalence_ratio'].max() * 1.1))
        axes[i].grid(True, alpha=0.3)
        
        # Annotate statistically significant groups
        for j, (_, row) in enumerate(factor_data.iterrows()):
            if row['p_value'] < 0.05:
                stars = "**" if row['p_value'] < 0.01 else "*"
                axes[i].text(j, 0.1, stars, ha='center', va='bottom', fontsize=16, color='red')
                
        # Rotate labels if needed
        if len(factor_data) > 5 or max(len(str(x)) for x in factor_data[factor]) > 10:
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.savefig(f'Results/demographic_risk_{target_icd_code}.png')
    plt.close()

def plot_threshold_curves(results_df, target_icd_code, demographic_group=None):
    """
    Plot evaluation metrics across different decision thresholds.
    
    Args:
        results_df: DataFrame with threshold evaluation results
        target_icd_code: The ICD code being predicted
        demographic_group: Optional tuple of (column, value) for group-specific analysis
    """
    plt.figure(figsize=(12, 10))
    
    # Standard metrics plot
    plt.subplot(3, 1, 1)
    plt.plot(results_df['threshold'], results_df['sensitivity'], 'b-', label='Sensitivity/Recall')
    plt.plot(results_df['threshold'], results_df['specificity'], 'r-', label='Specificity')
    plt.plot(results_df['threshold'], results_df['precision'], 'g-', label='Precision')
    plt.plot(results_df['threshold'], results_df['f1'], 'm-', label='F1 Score')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    
    # Error counts plot
    plt.subplot(3, 1, 2)
    plt.plot(results_df['threshold'], results_df['false_negatives'], 'r-', label='False Negatives')
    plt.plot(results_df['threshold'], results_df['false_positives'], 'b-', label='False Positives')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Count')
    plt.title('Errors vs. Threshold')
    plt.legend()
    plt.grid(True)
    
    # Demographic comparison (if available)
    if demographic_group is None and any(col.endswith('_fnr') for col in results_df.columns):
        plt.subplot(3, 1, 3)
        
        # Find columns with FNR metrics for demographic groups
        fnr_cols = [col for col in results_df.columns if col.endswith('_fnr')]
        
        # Plot FNR by demographic group
        for col in fnr_cols:
            group_name = col.replace('_fnr', '')
            plt.plot(results_df['threshold'], results_df[col], 
                    label=f'{group_name} FNR')
        
        plt.xlabel('Decision Threshold')
        plt.ylabel('False Negative Rate')
        plt.title('False Negative Rates by Demographic Group')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()

def plot_roc_comparison(results_without_demographics, results_with_demographics, target_icd_code):
    """
    Plot ROC curves comparing models with and without demographics.
    
    Args:
        results_without_demographics: Results dictionary for model without demographics
        results_with_demographics: Results dictionary for model with demographics
        target_icd_code: The ICD code being predicted
    """
    plt.figure(figsize=(10, 6))
    
    # Without demographics
    fpr1, tpr1, _ = roc_curve(
        results_without_demographics['true_labels'], 
        results_without_demographics['predictions'])
    
    plt.plot(fpr1, tpr1, 'b-', 
             label=f'Without Demographics (AUC = {results_without_demographics["auc_roc"]:.3f})')
    
    # With demographics
    fpr2, tpr2, _ = roc_curve(
        results_with_demographics['true_labels'], 
        results_with_demographics['predictions'])
    
    plt.plot(fpr2, tpr2, 'r-', 
             label=f'With Demographics (AUC = {results_with_demographics["auc_roc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison for {target_icd_code}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'Results/roc_comparison_{target_icd_code}.png')
    plt.close()
    
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
    
    plt.savefig(f'Results/fairness_comparison_{target_icd_code}.png')
    plt.close()

def plot_disparity_heatmap(disparity_metrics, target_icd_code):
    """
    Creates a heatmap visualization of disparities across demographic variables.
    
    Args:
        disparity_metrics: Dictionary from calculate_disparity_metrics
        target_icd_code: The ICD code being analyzed
    """
    # Extract metrics for visualization
    demo_vars = list(disparity_metrics.keys())
    
    if not demo_vars:
        return
        
    # Create a DataFrame for the heatmap
    metrics_to_plot = ['fnr_disparity', 'sensitivity_disparity', 'equal_opportunity_diff', 'disparate_impact']
    labels = ['FNR Disparity', 'Sensitivity Disparity', 'Equal Opportunity Diff', 'Disparate Impact']
    
    data = []
    for demo_var in demo_vars:
        row = [disparity_metrics[demo_var].get(metric, np.nan) for metric in metrics_to_plot]
        data.append(row)
    
    df = pd.DataFrame(data, index=demo_vars, columns=labels)
    
    # Create heatmap
    plt.figure(figsize=(10, len(demo_vars) * 1.5 + 2))
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    sns.heatmap(df, annot=True, cmap=cmap, center=1.0 if 'Disparate Impact' in df.columns else 0,
               linewidths=0.5, fmt='.2f')
    
    plt.title(f'Fairness Disparity Metrics for {target_icd_code}')
    plt.tight_layout()
    plt.savefig(f'Results/disparity_heatmap_{target_icd_code}.png')
    plt.close()