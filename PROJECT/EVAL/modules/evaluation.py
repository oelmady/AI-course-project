import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    confusion_matrix, 
    classification_report,
    accuracy_score, 
    f1_score
)
from joblib import Parallel, delayed
import multiprocessing

from .visualization import (
    plot_evaluation_metrics, 
    plot_threshold_curves, 
    plot_roc_comparison
)
from .model_builder import (
    build_predictor, 
    predict_likelihood,
    process_patient_batch
)
from .demographic_analyzer import (
    analyze_demographic_error_patterns,
    calculate_demographic_metrics
)
from .fairness import calculate_disparity_metrics
from .visualization import plot_fairness_comparison, plot_disparity_heatmap
    
    
def calculate_model_metrics(true_labels, predictions, decision_threshold=0.5):
    """Calculate all relevant metrics from predictions and labels"""
    binary_predictions = np.array(predictions) >= decision_threshold
    
    metrics = {
        'auc_roc': roc_auc_score(true_labels, predictions),
        'accuracy': accuracy_score(true_labels, binary_predictions),
        'f1': f1_score(true_labels, binary_predictions)
    }
    
    # Add precision-recall metrics
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    metrics['pr_auc'] = auc(recall, precision)
    
    # Add confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(true_labels, binary_predictions).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['false_negatives'] = fn
    metrics['false_positives'] = fp
    
    return metrics

def optimized_parallel_predictions(test_patients, G, k, parallel_jobs):
    """More efficient parallel prediction"""
    # Create balanced batches
    batch_size = max(1, len(test_patients) // parallel_jobs)
    batches = [test_patients[i:i + batch_size] for i in range(0, len(test_patients), batch_size)]
    
    # Process in parallel with shared memory for graph
    with Parallel(n_jobs=parallel_jobs, prefer="threads") as parallel:
        results = parallel(delayed(process_patient_batch)(batch, G, k) 
                           for batch in batches)
    
    # More efficient dictionary construction
    predictions_dict = {pid: prob for batch_result in results 
                       for pid, prob in batch_result}
    
    return [predictions_dict.get(pid, 0.0) for pid in test_patients]
    
    
def evaluate_threshold_curve_with_demographics(diagnoses_df, 
                                             demographics_df,
                                             target_icd_code, 
                                             k=11, 
                                             sample_size=2000, 
                                             class_weight=3.0,
                                             include_demographics=True,
                                             demographic_weight=0.5,
                                             demographic_group=None):
    """
    Evaluates model performance across different decision thresholds
    with demographic subgroup analysis to study SDOH impact on errors.
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        demographics_df: DataFrame with demographic data
        target_icd_code: The ICD code to predict
        k: Number of nearest neighbors
        sample_size: Limit on patients to include
        class_weight: Weight for target class in prediction
        include_demographics: Whether to include demographic features
        demographic_weight: Weight for demographic features
        demographic_group: Optional tuple of (column, value) to analyze a specific group
        
    Returns:
        DataFrame with threshold evaluation results
    """
    print(f"Evaluating threshold curve for {target_icd_code} with demographic analysis...")
    
    # Build the model with demographics
    G, target_patients, non_target_patients = build_predictor(
        diagnoses_df, 
        demographics_df,
        target_icd_code, 
        sample_size=sample_size,
        class_weight=class_weight,
        include_demographics=include_demographics,
        demographic_weight=demographic_weight
    )
    
    # Prepare data
    all_patients = target_patients + non_target_patients
    all_labels = [1] * len(target_patients) + [0] * len(non_target_patients)
    
    # Make predictions
    predictions_with_demographics = []
    for patient_id in all_patients:
        likelihood = predict_likelihood(G, patient_id, k)
        # Store patient ID with prediction and true label
        predictions_with_demographics.append({
            'subject_id': patient_id,
            'prediction': likelihood,
            'true_label': 1 if patient_id in target_patients else 0
        })
    
    # Convert to DataFrame for easier analysis
    predictions_df = pd.DataFrame(predictions_with_demographics)
    
    # If analyzing a specific demographic group, merge with demographics and filter
    if demographic_group is not None and isinstance(demographic_group, tuple) and len(demographic_group) == 2:
        column, value = demographic_group
        if column in demographics_df.columns:
            # Merge predictions with demographics
            predictions_df = predictions_df.merge(demographics_df, on='subject_id', how='inner')
            # Filter for the specific demographic group
            predictions_df = predictions_df[predictions_df[column] == value]
            print(f"Analyzing threshold curve for demographic group: {column}={value}")
            print(f"Group size: {len(predictions_df)} patients")
    
    # If we need general demographic analysis, merge with demographics
    elif demographic_group is None and include_demographics:
        predictions_df = predictions_df.merge(demographics_df, on='subject_id', how='inner')
    
    # Evaluate across thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    demographic_columns = ['gender', 'race', 'insurance'] if demographic_group is None else []
    
    for threshold in thresholds:
        # Calculate metrics for this threshold
        result = calculate_demographic_metrics(predictions_df, threshold, demographic_columns)
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\nPerformance across decision thresholds:")
    # Set display options to show the entire table
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None,
                           'display.width', None,
                           'display.precision', 3):
        print(results_df[['threshold', 'sensitivity', 'specificity', 'precision', 'f1', 
                          'false_negatives', 'false_positives']])
    
    # Plot the results
    plot_threshold_curves(results_df, target_icd_code, demographic_group)
    
    return results_df

def diagnosis_evaluation_with_demographics(
    diagnoses_df,
    demographics_df,
    target_icd_code, 
    k=11,
    n_folds=5,
    similarity_threshold=0.15,
    decision_threshold=0.3,
    sample_size=2000,
    parallel_jobs=-1,
    use_matrix_format=True,
    class_weight=3.0,
    demographic_weight=0.5,
    include_demographics=True):
    """
    Comprehensive evaluation of the KNN comorbidity prediction model with demographic analysis.
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        demographics_df: DataFrame with demographic data
        target_icd_code: The ICD code to predict
        k: Number of nearest neighbors
        n_folds: Number of folds for cross-validation
        similarity_threshold: Threshold for creating graph edges
        decision_threshold: Threshold for binary classification
        sample_size: Optional limit on patients to include (None for all)
        parallel_jobs: Number of parallel jobs (-1 for all available cores)
        use_matrix_format: Whether to use efficient matrix format for large datasets
        class_weight: Weight for target class in prediction
        demographic_weight: Weight applied to demographic features in similarity calculation
        include_demographics: Whether to include demographic features
        
    Returns:
        results_dict: Dictionary with evaluation metrics
    """
    print(f"\n=== Comprehensive Evaluation with Demographics for ICD code: {target_icd_code} ===")
    
    
    # Build the model with demographics
    G, target_patients, non_target_patients = build_predictor(
        diagnoses_df, 
        demographics_df,
        target_icd_code, 
        similarity_threshold=similarity_threshold,
        sample_size=sample_size,
        use_matrix_format=use_matrix_format,
        class_weight=class_weight,
        include_demographics=include_demographics,
        demographic_weight=demographic_weight
    )
    
    # Prepare data for cross-validation
    all_patients = target_patients + non_target_patients
    all_labels = [1] * len(target_patients) + [0] * len(non_target_patients)
    
    # Convert to numpy arrays
    patients_array = np.array(all_patients)
    labels_array = np.array(all_labels)
    
    # Initialize metrics storage
    all_predictions = []
    all_true_labels = []
    cv_auc_scores = []
    cv_pr_auc_scores = []
    cv_accuracy = []
    cv_f1_scores = []
    
    # Track patient IDs for demographic analysis
    prediction_patient_ids = []
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Configure parallel processing
    if parallel_jobs == -1:
        parallel_jobs = multiprocessing.cpu_count()
    
    fold = 1
    # Perform cross-validation
    for _, test_idx in skf.split(patients_array, labels_array):
        
        # Get test patients and labels for this fold
        test_patients = patients_array[test_idx].tolist()
        test_labels = labels_array[test_idx].tolist()
        
        # Make predictions using parallel processing for large test sets
        if len(test_patients) > 5000 and parallel_jobs > 1:
            print(f"  Processing predictions in parallel with {parallel_jobs} workers...")
            predictions = optimized_parallel_predictions(test_patients, G, k, parallel_jobs)
        else:
            predictions = []
            for patient_id in test_patients:
                likelihood = predict_likelihood(G, patient_id, k)
                predictions.append(likelihood)
        
        # Convert likelihoods to binary predictions
        binary_predictions = np.array(all_predictions) >= decision_threshold
        
        # Calculate metrics
        auc_score = roc_auc_score(test_labels, predictions)
        precision, recall, _ = precision_recall_curve(test_labels, predictions)
        pr_auc = auc(recall, precision)
        accuracy = accuracy_score(test_labels, binary_predictions)
        f1 = f1_score(test_labels, binary_predictions)
        
        # Store metrics
        cv_auc_scores.append(auc_score)
        cv_pr_auc_scores.append(pr_auc)
        cv_accuracy.append(accuracy)
        cv_f1_scores.append(f1)
        
        # Store predictions and labels for later analysis
        all_predictions.extend(predictions)
        all_true_labels.extend(test_labels)
        prediction_patient_ids.extend(test_patients)
        
        fold += 1
    
    # Calculate overall metrics
    print("\n=== Overall Results ===")
    print(f"Mean AUC-ROC: {np.mean(cv_auc_scores):.3f} ± {np.std(cv_auc_scores):.3f}")
    print(f"Mean PR-AUC: {np.mean(cv_pr_auc_scores):.3f} ± {np.std(cv_pr_auc_scores):.3f}")
    print(f"Mean Accuracy: {np.mean(cv_accuracy):.3f} ± {np.std(cv_accuracy):.3f}")
    print(f"Mean F1 Score: {np.mean(cv_f1_scores):.3f} ± {np.std(cv_f1_scores):.3f}")
    
    # Create binary predictions for classification report
    binary_predictions = np.array(all_predictions) >= decision_threshold
    
    # Return results in a dictionary
    results = {
        'auc_roc': np.mean(cv_auc_scores),
        'pr_auc': np.mean(cv_pr_auc_scores),
        'accuracy': np.mean(cv_accuracy),
        'f1_score': np.mean(cv_f1_scores),
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'patient_ids': prediction_patient_ids,
        'binary_predictions': binary_predictions
    }
    
    # Plot evaluation metrics and analyze results
    analyze_and_visualize_results(results, diagnoses_df, demographics_df, target_icd_code, prediction_patient_ids)
    
    return results

def analyze_and_visualize_results(results, diagnoses_df, demographics_df, target_icd_code, patient_ids=None):
    """
    Analyzes and visualizes prediction results, including evaluation metrics, 
    classification report, predictive comorbidities, and demographic error patterns.
    
    Args:
        results: Dictionary containing prediction results 
        diagnoses_df: DataFrame with diagnoses data
        demographics_df: DataFrame with demographic data
        target_icd_code: The ICD code being predicted
        patient_ids: List of patient IDs (if not included in results)
    """
    # Extract data from results dictionary
    all_true_labels = results['true_labels']
    all_predictions = results['predictions'] 
    binary_predictions = results['binary_predictions']
    prediction_patient_ids = results.get('patient_ids', patient_ids)
    
    # Plot evaluation metrics
    plot_evaluation_metrics(all_true_labels, all_predictions, binary_predictions, target_icd_code)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, binary_predictions))
    
    # Analyze demographic error patterns
    if demographics_df is not None and prediction_patient_ids is not None:
        print("\nAnalyzing error patterns across demographic groups...")
        demographic_error_analysis = analyze_demographic_error_patterns(
            all_predictions, all_true_labels, prediction_patient_ids, demographics_df
        )
        # Visualization is called inside analyze_demographic_error_patterns

def compare_models_with_without_demographics(diagnoses_df, demographics_df, target_icd_code, **kwargs):
    """
    Compares performance of models with and without demographic features.
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        demographics_df: DataFrame with demographic data
        target_icd_code: The ICD code to predict
        **kwargs: Additional parameters for evaluation
        
    Returns:
        Tuple of results for both models
    """
    
    print(f"\n=== Comparing Models With and Without Demographics for {target_icd_code} ===\n")
    
    # Run model without demographics
    print("\n** Running Model WITHOUT Demographics **")
    results_without_demographics = diagnosis_evaluation_with_demographics(
        diagnoses_df, demographics_df, target_icd_code, 
        include_demographics=False, **kwargs
    )
    
    # Run model with demographics
    print("\n** Running Model WITH Demographics **")
    results_with_demographics = diagnosis_evaluation_with_demographics(
        diagnoses_df, demographics_df, target_icd_code,
        include_demographics=True, **kwargs
    )
    
    # Compare results
    print("\n=== Performance Comparison ===")
    metrics = ['auc_roc', 'pr_auc', 'accuracy', 'f1_score']
    
    for metric in metrics:
        without_demo = results_without_demographics[metric]
        with_demo = results_with_demographics[metric]
        diff = with_demo - without_demo
        print(f"{metric.upper()}: Without demographics: {without_demo:.3f}, With demographics: {with_demo:.3f}")
        print(f"Difference: {diff:.3f} ({'+' if diff > 0 else ''}{diff/without_demo*100:.1f}%)")
    
    # Plot comparison of ROC curves
    plot_roc_comparison(results_without_demographics, results_with_demographics, target_icd_code)
    
    return results_without_demographics, results_with_demographics

def analyze_demographic_group(diagnoses_df, demographics_df, target_icd_code, demo_var, group, sample_size):
    """Analyze a single demographic group for fairness evaluation"""
    print(f"\nEvaluating model for {demo_var}={group}")
    
    # Get patients in this demographic group
    group_patients = demographics_df[demographics_df[demo_var] == group]['subject_id']
    
    # Filter diagnoses to only these patients
    group_diagnoses = diagnoses_df[diagnoses_df['subject_id'].isin(group_patients)]
    
    # Skip if too few patients
    if len(group_diagnoses['subject_id'].unique()) < 100:
        print(f"  Too few patients with {demo_var}={group}, skipping")
        return None
        
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
    
    # Store optimal threshold results
    optimal_threshold = threshold_results.loc[threshold_results['f1'].idxmax()]
    result = {
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
    
    return (group, result)

def evaluate_demographic_fairness(diagnoses_df, demographics_df, target_icd_code, sample_size=3000):
    """
    Evaluates model fairness across demographic groups.
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        demographics_df: DataFrame with demographic data
        target_icd_code: The ICD code to analyze
        sample_size: Sample size for each group's evaluation
        
    Returns:
        Dictionary with fairness results by demographic group
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
    
    # Results storage
    results = {}
    
    # Analyze major demographic variables
    for demo_var in ['race', 'gender', 'insurance']:
        if demo_var not in demographics_df.columns:
            continue
            
        # Get largest groups covering at least 90% of patients
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
        
        # Evaluate each group
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
            
            # Store optimal threshold results
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
    
    # Calculate and display disparity metrics
    disparity_metrics = calculate_disparity_metrics(results)
    
    for demo_var, metrics in disparity_metrics.items():
        print(f"\nDisparities for {demo_var.upper()}:")
        print(f"  False Negative Rate Disparity: {metrics['fnr_disparity']:.2f}x")
        print(f"  Sensitivity Disparity: {metrics['sensitivity_disparity']:.2f}x")
        
        # Flag significant disparities
        if metrics['significant_disparity']:
            print(f"  ⚠️ Warning: Significant disparity detected across {demo_var} groups")
            print(f"     Group most affected by false negatives: {metrics['most_affected_fnr']}")
            print(f"     Group most affected by missed cases: {metrics['most_affected_sens']}")
    
    # Visualize fairness metrics
    plot_fairness_comparison(results, target_icd_code)
    plot_disparity_heatmap(disparity_metrics, target_icd_code)
    
    return results, disparity_metrics