import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import chi2    
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve, 
    auc,
    confusion_matrix, 
    classification_report,
    accuracy_score, 
    f1_score)
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

from knn_diagnosis_eval import (plot_evaluation_metrics, 
                                predict_likelihood, 
                                process_patient_batch, 
                                analyze_predictive_comorbidities)

def build_predictor(diagnoses_df, 
                   demographics_df=None,
                   target_icd_code=None, 
                   similarity_threshold=0.2,
                   sample_size=2000,
                   use_matrix_format=True, 
                   class_weight=3.0,
                   include_demographics=True,
                   demographic_weight=0.5):
    """
    Builds a KNN-based comorbidity prediction model incorporating demographic features.
    
    Args:
        diagnoses_df: DataFrame containing diagnosis data
        demographics_df: DataFrame containing demographic data (optional)
        target_icd_code: The ICD code to predict
        similarity_threshold: Minimum similarity to create a connection
        sample_size: Optional limit on number of patients to include (None for all)
        use_matrix_format: Whether to use efficient matrix format for large datasets
        class_weight: Weight for target class in prediction
        include_demographics: Whether to include demographic features
        demographic_weight: Weight applied to demographic features in similarity calculation
        
    Returns:
        G: NetworkX graph of patient similarities
        target_patients: List of patients with the target diagnosis
        non_target_patients: List of patients without the target diagnosis
    """
    print(f"Building prediction model for ICD code: {target_icd_code}")
    if include_demographics and demographics_df is not None:
        print("Including demographic features in similarity calculation")
    
    # Step 1: Create patient-diagnosis mapping
    patient_diagnoses = diagnoses_df.groupby('subject_id')['icd_code'].apply(set).reset_index()
    
    # Identify patients with and without the target diagnosis
    target_patients = set(diagnoses_df[diagnoses_df['icd_code'] == target_icd_code]['subject_id'].unique())
    all_patients = set(diagnoses_df['subject_id'].unique())
    non_target_patients = all_patients - target_patients
    
    print(f"Patients with {target_icd_code} diagnosis: {len(target_patients)}")
    print(f"Patients without {target_icd_code} diagnosis: {len(non_target_patients)}")
    
    # For large datasets, we can work with a representative sample
    if sample_size and len(patient_diagnoses) > sample_size:
        # Maintain class proportions in the sample
        target_ratio = len(target_patients) / len(all_patients)
        target_sample_size = int(sample_size * target_ratio)
        non_target_sample_size = sample_size - target_sample_size
        
        # Sample from each group
        target_sample = list(target_patients)
        non_target_sample = list(non_target_patients)
        
        np.random.shuffle(target_sample)
        np.random.shuffle(non_target_sample)
        
        target_sample = target_sample[:target_sample_size]
        non_target_sample = non_target_sample[:non_target_sample_size]
        
        sample_patients = set(target_sample + non_target_sample)
        patient_diagnoses = patient_diagnoses[patient_diagnoses['subject_id'].isin(sample_patients)]
        
        # Update target and non_target patient lists
        target_patients = set(target_sample)
        non_target_patients = set(non_target_sample)
        
        print(f"Using a representative sample of {len(patient_diagnoses)} patients")
    
    # Step 2: Incorporate demographics if requested
    patient_ids = patient_diagnoses['subject_id'].tolist()
    
    if include_demographics and demographics_df is not None:
        print("Processing demographic features...")
        # Get demographics for patients in our dataset
        demo_features = demographics_df[demographics_df['subject_id'].isin(patient_ids)].copy()
        
        # Ensure all patients have demographic records
        patient_with_demos = set(demo_features['subject_id'])
        patients_without_demos = set(patient_ids) - patient_with_demos
        
        if patients_without_demos:
            print(f"Warning: {len(patients_without_demos)} patients don't have demographic data")
            # Keep only patients with demographic data
            patient_diagnoses = patient_diagnoses[patient_diagnoses['subject_id'].isin(patient_with_demos)]
            target_patients = target_patients.intersection(patient_with_demos)
            non_target_patients = non_target_patients.intersection(patient_with_demos)
            patient_ids = [pid for pid in patient_ids if pid in patient_with_demos]
        
        # One-hot encode categorical variables
        categorical_cols = ['gender', 'insurance', 'race', 'language', 'marital_status']
        # Only include columns that actually exist in the DataFrame
        cols_to_encode = [col for col in categorical_cols if col in demo_features.columns]
        
        if cols_to_encode:
            demo_encoded = pd.get_dummies(demo_features, columns=cols_to_encode, drop_first=False)
            
            # Remove the subject_id column for feature matrix
            demo_features = demo_encoded.drop('subject_id', axis=1)
            
            # Create a mapping from patient_id to demographics feature vector
            demographics_map = {row['subject_id']: row.drop('subject_id').values 
                               for _, row in demo_encoded.iterrows()}
        else:
            print("Warning: No categorical demographic features found")
            include_demographics = False
    
    # Step 3: Convert diagnoses to binary indicators
    mlb = MultiLabelBinarizer(sparse_output=use_matrix_format)
    diagnoses_matrix = mlb.fit_transform(patient_diagnoses['icd_code'])
    
    # Remove the target diagnosis from the feature set to prevent data leakage
    target_feature_idx = -1
    for i, feature in enumerate(mlb.classes_):
        if feature == target_icd_code:
            target_feature_idx = i
            break
    
    # Handle matrix data appropriately based on format
    if use_matrix_format:
        # For sparse matrices, convert to lil_matrix for efficient modification
        X = diagnoses_matrix.copy()
        if target_feature_idx >= 0:
            X = X.tolil()
            X[:, target_feature_idx] = 0
            # Convert back to CSR for efficient computation
            X = X.tocsr()
    else:
        # For dense matrices, filter column
        if target_feature_idx >= 0:
            mask = np.ones(len(mlb.classes_), dtype=bool)
            mask[target_feature_idx] = False
            X = diagnoses_matrix[:, mask]
        else:
            X = diagnoses_matrix
    
    # Step 4: Calculate patient similarities
    print("Computing patient similarities...")
    similarity_matrix = cosine_similarity(X, dense_output=False if use_matrix_format else True)
    
    # Step 5: Create a graph representation
    print("Building patient similarity network...")
    G = nx.Graph()
    
    # Add nodes (patients)
    for i, patient_id in enumerate(patient_ids):
        has_target = patient_id in target_patients
        node_weight = class_weight if has_target else 1.0
        
        # Add demographics as node attributes if available
        if include_demographics and demographics_df is not None and cols_to_encode:
            G.add_node(patient_id, has_target=has_target, weight=node_weight, 
                      demographics=demographics_map.get(patient_id, None))
        else:
            G.add_node(patient_id, has_target=has_target, weight=node_weight)
    
    # Add edges (significant similarities between patients)
    edge_count = 0
    if use_matrix_format:
        # Efficiently process sparse similarity matrix
        cx = similarity_matrix.tocoo()
        for i, j, sim in zip(cx.row, cx.col, cx.data):
            if i < j and sim > similarity_threshold:  
                # Avoid duplicates and self-loops
                # If demographics are included, adjust similarity by demographic similarity
                if include_demographics and demographics_df is not None and cols_to_encode:
                    patient1_id = patient_ids[i]
                    patient2_id = patient_ids[j]
                    
                    # Get demographic vectors for both patients
                    demo1 = demographics_map.get(patient1_id)
                    demo2 = demographics_map.get(patient2_id)
                    
                    if demo1 is not None and demo2 is not None:
                        # Calculate demographic similarity (cosine similarity)
                        demo_sim = np.dot(demo1, demo2) / (np.linalg.norm(demo1) * np.linalg.norm(demo2))
                        
                        # Weighted combination of diagnosis and demographic similarity
                        combined_sim = (1 - demographic_weight) * sim + demographic_weight * demo_sim
                        G.add_edge(patient_ids[i], patient_ids[j], weight=combined_sim)
                        edge_count += 1
                else:
                    G.add_edge(patient_ids[i], patient_ids[j], weight=sim)
                    edge_count += 1
    else:
        # Process dense similarity matrix
        rows, cols = np.where(similarity_matrix > similarity_threshold)
        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicates and self-loops
                # Similar demographic adjustment as above
                if include_demographics and demographics_df is not None and cols_to_encode:
                    patient1_id = patient_ids[i]
                    patient2_id = patient_ids[j]
                    
                    demo1 = demographics_map.get(patient1_id)
                    demo2 = demographics_map.get(patient2_id)
                    
                    if demo1 is not None and demo2 is not None:
                        demo_sim = np.dot(demo1, demo2) / (np.linalg.norm(demo1) * np.linalg.norm(demo2))
                        combined_sim = (1 - demographic_weight) * similarity_matrix[i, j] + demographic_weight * demo_sim
                        G.add_edge(patient_ids[i], patient_ids[j], weight=combined_sim)
                        edge_count += 1
                else:
                    G.add_edge(patient_ids[i], patient_ids[j], weight=similarity_matrix[i, j])
                    edge_count += 1
    
    print(f"Network built with {G.number_of_nodes()} patients and {edge_count} connections")
    
    return G, list(target_patients), list(non_target_patients)


def load_demographics_data(dataset_path="Data"):
    """Load demographics data and return as DataFrame"""
    demographics_path = os.path.join(dataset_path, "demographics.csv")
    if os.path.exists(demographics_path):
        return pd.read_csv(demographics_path)
    else:
        raise FileNotFoundError(f"Demographics file not found at {demographics_path}")
    
def plot_demographic_error_analysis(error_analysis):
    """
    Creates visualizations of error rates across demographic groups.
    
    Args:
        error_analysis: Dictionary with error analysis by demographic factor
    """
    n_factors = len(error_analysis)
    fig, axes = plt.subplots(n_factors, 1, figsize=(12, 4*n_factors), constrained_layout=True)
    
    if n_factors == 1:
        axes = [axes]  # Make iterable for consistent access
    
    for i, (factor, data) in enumerate(error_analysis.items()):
        # Sort by error rate for better visualization
        data = data.sort_values('error_rate', ascending=False)
        
        # Plot error rates with bar width proportional to group size
        sizes = data['count'] / data['count'].max() * 0.8 + 0.2  # Scale between 0.2 and 1.0
        bars = axes[i].bar(data[factor], data['error_rate'], width=sizes, alpha=0.7)
        
        # Add count labels
        for bar, count in zip(bars, data['count']):
            axes[i].text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + 0.01, 
                        str(count), 
                        ha='center', va='bottom', 
                        fontsize=8)
        
        axes[i].set_title(f'Error Rate by {factor.title()}')
        axes[i].set_ylabel('Error Rate')
        axes[i].set_ylim(0, min(1.0, data['error_rate'].max() * 1.2))  # Set sensible y limit
        
        # Rotate x-labels if needed
        if len(data) > 5 or max(len(str(x)) for x in data[factor]) > 10:
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.savefig('Results/demographic_error_analysis.png')

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
        'error': [1 if abs(p - t) > 0.5 else 0 for p, t in zip(all_predictions, all_true_labels)]
    })
    
    # Merge with demographics
    analysis_df = results_df.merge(demographics_df, on='subject_id', how='inner')
    
    # Calculate error rates by demographic groups
    demographic_factors = ['gender', 'insurance', 'race', 'language', 'marital_status']
    error_analysis = {}
    
    for factor in demographic_factors:
        if factor in demographics_df.columns:
            group_error = analysis_df.groupby(factor)['error'].mean().reset_index()
            group_error.columns = [factor, 'error_rate']
            group_count = analysis_df.groupby(factor)['subject_id'].count().reset_index()
            group_count.columns = [factor, 'count']
            
            # Merge error rate with count
            group_stats = group_error.merge(group_count, on=factor)
            error_analysis[factor] = group_stats
    
    # Visualize error rates by demographic groups
    plot_demographic_error_analysis(error_analysis)
    
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
    
    # Analyze each demographic factor
    demographic_factors = ['gender', 'insurance', 'race', 'language', 'marital_status']
    result_tables = []
    
    for factor in demographic_factors:
        if factor in demographics_with_target.columns:
            # Create contingency table
            contingency = pd.crosstab(
                demographics_with_target[factor],
                demographics_with_target['has_target'],
                normalize='columns'
            )
            
            # Add row counts
            factor_counts = demographics_with_target[factor].value_counts()
            contingency['count'] = factor_counts
            
            # Calculate prevalence ratio
            contingency['prevalence_ratio'] = contingency[1] / contingency[0]
            
            # Reset index to make factor a column
            contingency = contingency.reset_index()
            
            # Add factor name as column
            contingency['factor'] = factor
            
            result_tables.append(contingency)
    
    # Combine all results
    all_results = pd.concat(result_tables, ignore_index=True)
    
    # Display results
    print(f"\nDemographic Risk Factor Analysis for {target_icd_code}:")
    for factor in demographic_factors:
        if factor in demographics_with_target.columns:
            factor_results = all_results[all_results['factor'] == factor]
            print(f"\n{factor.upper()} as a risk factor:")
            for _, row in factor_results.sort_values('prevalence_ratio', ascending=False).iterrows():
                print(f"  {row[factor]}: {row[1]*100:.2f}% (vs {row[0]*100:.2f}%), Ratio: {row['prevalence_ratio']:.2f}x, Count: {row['count']}")
    
    return all_results

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
            # Split test patients into batches for parallel processing
            batch_size = max(1, len(test_patients) // parallel_jobs)
            batches = [test_patients[i:i + batch_size] for i in range(0, len(test_patients), batch_size)]
            
            print(f"  Processing predictions across {len(batches)} batches...")
            # Process predictions in parallel
            results = Parallel(n_jobs=parallel_jobs)(
                delayed(process_patient_batch)(batch, G, k) for batch in batches
            )
            
            # Flatten results
            predictions_dict = {}
            for batch_result in results:
                for patient_id, likelihood in batch_result:
                    predictions_dict[patient_id] = likelihood
            
            # Ensure predictions are in the same order as test_patients
            predictions = [predictions_dict.get(patient_id, 0.0) for patient_id in test_patients]
        else:
            predictions = []
            for patient_id in test_patients:
                likelihood = predict_likelihood(G, patient_id, k)
                predictions.append(likelihood)
        
        # Convert likelihoods to binary predictions
        binary_predictions = [1 if p >= decision_threshold else 0 for p in predictions]
        
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
    binary_predictions = [1 if p >= decision_threshold else 0 for p in all_predictions]
    
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
    plt.figure(figsize=(10, 6))
    
    # Without demographics
    fpr1, tpr1, _ = roc_curve(results_without_demographics['true_labels'], 
                            results_without_demographics['predictions'])
    plt.plot(fpr1, tpr1, 'b-', 
             label=f'Without Demographics (AUC = {results_without_demographics["auc_roc"]:.3f})')
    
    # With demographics
    fpr2, tpr2, _ = roc_curve(results_with_demographics['true_labels'], 
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
    
    return results_without_demographics, results_with_demographics

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
    
    for threshold in thresholds:
        predictions_df['binary_pred'] = predictions_df['prediction'].apply(lambda p: 1 if p >= threshold else 0)
        
        # Calculate overall metrics
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
        
        # If we're doing demographic analysis without specific filter
        if demographic_group is None and 'gender' in predictions_df.columns:
            # Calculate metrics for each gender
            for demo_col in ['gender', 'race', 'insurance']:
                if demo_col in predictions_df.columns:
                    for group_value in predictions_df[demo_col].unique():
                        group_df = predictions_df[predictions_df[demo_col] == group_value]
                        # Only include groups with sufficient patients
                        if len(group_df) >= 20:  
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
        
        results.append(result)
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\nPerformance across decision thresholds:")
    # Set display options to show the entire table
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None,
                           'display.width', None,
                           'display.precision', 3):
        print(results_df[['threshold', 'sensitivity', 'specificity', 'precision', 'f1', 
                          'false_negatives', 'false_positives']])
    
    # Plot metrics vs threshold
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
    if demographic_group is None and 'gender_M_fnr' in results_df.columns:
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
    plt.savefig(f'Results/threshold_curves_{target_icd_code}.png')
    plt.show()
    
    return results_df

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
    
    # Visualize error rates
    plot_demographic_error_analysis(error_analysis)
    
    return error_analysis


if __name__ == "__main__":
    diagnoses_path = "Data/diagnoses_icd.csv"
    
    if os.path.exists(diagnoses_path):
        diagnoses_df = pd.read_csv(diagnoses_path)
    else:
        raise FileNotFoundError(f"Diagnoses file not found at {diagnoses_path}")
    
    # Load demographics data
    demographics_df = load_demographics_data()
    
    # Analyze demographic risk factors
    analyze_demographic_risk_factors(diagnoses_df, demographics_df, "I10")
    
    # Example: Evaluate prediction model for hypertension (code I10)
    diagnosis_evaluation_with_demographics(
        diagnoses_df,
        demographics_df,
        "I10",                  # Hypertension
        k=11,                   
        n_folds=5,              
        similarity_threshold=0.15,
        sample_size=2000,       
        parallel_jobs=-1
    )
    