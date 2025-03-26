import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                           confusion_matrix, classification_report,
                           accuracy_score, f1_score)
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import multiprocessing

'''
Implements KNN-based comorbidity analysis using a graph representation approach.
- Uses cross-validation to get reliable performance estimates
- Calculates comprehensive metrics (AUC, PR-AUC, accuracy, F1)
- Handles high-dimensional data efficiently
- Parallelizes computation for faster analysis
- Identifies the most predictive comorbidities using chi-square tests
'''

def build_balanced_predictor(diagnoses_df, target_icd_code, similarity_threshold=0.2, 
                   sample_size=None, use_matrix_format=True, 
                   class_weight=3.0):
    """
    Builds a KNN-based comorbidity prediction model for large clinical datasets
    using a graph representation approach.
    
    Args:
        diagnoses_df: DataFrame containing diagnosis data
        target_icd_code: The ICD code to predict
        similarity_threshold: Minimum similarity to create a connection
        sample_size: Optional limit on number of patients to include (None for all)
        use_matrix_format: Whether to use efficient matrix format for large datasets
        
    Returns:
        G: NetworkX graph of patient similarities
        target_patients: List of patients with the target diagnosis
        non_target_patients: List of patients without the target diagnosis
    """
    print(f"Building prediction model for ICD code: {target_icd_code}")
    
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
    
    # Step 2: Convert diagnoses to binary indicators
    # Using sparse format for more efficient memory usage with large datasets
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
            # Convert to LIL format for efficient column modification
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
    
    # Step 3: Calculate patient similarities
    print("Computing patient similarities...")
    similarity_matrix = cosine_similarity(X, dense_output=False if use_matrix_format else True)
    patient_ids = patient_diagnoses['subject_id'].tolist()
    
    # Step 4: Create a graph representation
    print("Building patient similarity network...")
    G = nx.Graph()
    
    # Add nodes (patients)
    for i, patient_id in enumerate(patient_ids):
        has_target = patient_id in target_patients
        node_weight = class_weight if has_target else 1.0
        G.add_node(patient_id, has_target=has_target, weight=node_weight)
    
    # Add edges (significant similarities between patients)
    if use_matrix_format:
        # Efficiently process sparse similarity matrix
        cx = similarity_matrix.tocoo()
        for i, j, sim in zip(cx.row, cx.col, cx.data):
            if i < j and sim > similarity_threshold:  # Avoid duplicates and self-loops
                G.add_edge(patient_ids[i], patient_ids[j], weight=sim)
    else:
        # Process dense similarity matrix
        rows, cols = np.where(similarity_matrix > similarity_threshold)
        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicates and self-loops
                G.add_edge(patient_ids[i], patient_ids[j], weight=similarity_matrix[i, j])
    
    print(f"Network built with {G.number_of_nodes()} patients and {G.number_of_edges()} connections")
    
    return G, list(target_patients), list(non_target_patients)

def predict_likelihood(G, patient_id, k=10):
    """
    Predicts the likelihood of a patient having the target diagnosis
    based on their k nearest neighbors in the similarity graph.
    
    Args:
        G: NetworkX graph of patient similarities
        patient_id: ID of the patient to predict for
        k: Number of nearest neighbors to consider
        
    Returns:
        likelihood: Predicted likelihood of having the target diagnosis
    """
    if patient_id not in G:
        return 0.0
    
    # Get neighbors with their similarity weights
    neighbors = [(neighbor, G[patient_id][neighbor]['weight']) 
                for neighbor in G.neighbors(patient_id)]
    
    # Sort by similarity (descending) and take top k
    neighbors.sort(key=lambda x: x[1], reverse=True)
    top_k_neighbors = neighbors[:k]
    
    if not top_k_neighbors:
        return 0.0
    
    # Calculate weighted average of target diagnosis presence
    total_weight = sum(
        weight * G.nodes[neighbor].get('weight', 1.0) 
        for neighbor, weight in top_k_neighbors)
    
    weighted_sum = sum(
        G.nodes[neighbor]['has_target'] * weight * G.nodes[neighbor].get('weight', 1.0)
        for neighbor, weight in top_k_neighbors)
    
    
    if total_weight == 0:
        return 0.0
    
    likelihood = weighted_sum / total_weight
    return likelihood

def process_patient_batch(patient_batch, G, k):
    """Process a batch of patients in parallel"""
    results = []
    for patient_id in patient_batch:
        likelihood = predict_likelihood(G, patient_id, k)
        results.append((patient_id, likelihood))
    return results

def diagnosis_evaluation(diagnoses_df, target_icd_code, k=10, 
                              n_folds=5, similarity_threshold=0.1, 
                              decision_threshold=0.3, 
                              sample_size=None, 
                              parallel_jobs=-1,
                              use_matrix_format=True, 
                              class_weight=3.0):
    """
    Comprehensive evaluation of the KNN comorbidity prediction model.
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        target_icd_code: The ICD code to predict
        k: Number of nearest neighbors
        n_folds: Number of folds for cross-validation
        similarity_threshold: Threshold for creating graph edges
        decision_threshold: Threshold for binary classification
        sample_size: Optional limit on patients to include (None for all)
        parallel_jobs: Number of parallel jobs (-1 for all available cores)
        use_matrix_format: Whether to use efficient matrix format for large datasets
        class_weight: Weight for target class in prediction
        
    Returns:
        results_dict: Dictionary with evaluation metrics
    """
    print(f"\n=== Comprehensive Evaluation for ICD code: {target_icd_code} ===")
    
    # Build the model 
    G, target_patients, non_target_patients = build_balanced_predictor(
        diagnoses_df, target_icd_code, 
        similarity_threshold=similarity_threshold,
        sample_size=sample_size,
        use_matrix_format=use_matrix_format,
        class_weight=class_weight
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
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Configure parallel processing
    if parallel_jobs == -1:
        parallel_jobs = multiprocessing.cpu_count()
    
    fold = 1
    # Perform cross-validation
    for train_idx, test_idx in skf.split(patients_array, labels_array):
        print(f"\nFold {fold}/{n_folds}")
        
        # Get test patients and labels for this fold
        test_patients = patients_array[test_idx].tolist()
        test_labels = labels_array[test_idx].tolist()
        
        # Make predictions using parallel processing for large test sets
        if len(test_patients) > 1000 and parallel_jobs > 1:
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
            # For smaller datasets, sequential processing is sufficient
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
        
        print(f"  AUC-ROC: {auc_score:.3f}")
        print(f"  PR-AUC: {pr_auc:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        
        fold += 1
    
    # Calculate overall metrics
    print("\n=== Overall Results ===")
    print(f"Mean AUC-ROC: {np.mean(cv_auc_scores):.3f} ± {np.std(cv_auc_scores):.3f}")
    print(f"Mean PR-AUC: {np.mean(cv_pr_auc_scores):.3f} ± {np.std(cv_pr_auc_scores):.3f}")
    print(f"Mean Accuracy: {np.mean(cv_accuracy):.3f} ± {np.std(cv_accuracy):.3f}")
    print(f"Mean F1 Score: {np.mean(cv_f1_scores):.3f} ± {np.std(cv_f1_scores):.3f}")
    
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(all_true_labels, all_predictions):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {target_icd_code}')
    plt.legend()
    
    # Confusion Matrix
    binary_predictions = [1 if p >= decision_threshold else 0 for p in all_predictions]
    cm = confusion_matrix(all_true_labels, binary_predictions)
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f'evaluation_{target_icd_code}.png')
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, binary_predictions))
    
    # Analyze predictive comorbidities
    print("\nAnalyzing most significant comorbidities...")
    analyze_predictive_comorbidities(diagnoses_df, target_icd_code)
    
    # Return results in a dictionary
    results = {
        'auc_roc': np.mean(cv_auc_scores),
        'pr_auc': np.mean(cv_pr_auc_scores),
        'accuracy': np.mean(cv_accuracy),
        'f1_score': np.mean(cv_f1_scores),
        'predictions': all_predictions,
        'true_labels': all_true_labels
    }
    return results

def analyze_predictive_comorbidities(diagnoses_df, target_icd_code):
    """
    Identifies which comorbidities are most strongly associated with the target diagnosis.
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        target_icd_code: The ICD code being predicted
        
    Returns:
        DataFrame: Feature importance metrics for comorbidities
    """
    from sklearn.feature_selection import chi2
    
    print(f"Analyzing predictive comorbidities for ICD code: {target_icd_code}")
    
    # Get patients with the target diagnosis
    target_patients = diagnoses_df[diagnoses_df['icd_code'] == target_icd_code]['subject_id'].unique()
    target_patients_set = set(target_patients)
    print(f"Found {len(target_patients)} patients with {target_icd_code} diagnosis")
    
    # For each patient, get all diagnoses EXCEPT the target one
    all_diagnoses_except_target = diagnoses_df[diagnoses_df['icd_code'] != target_icd_code]
    
    # Now count occurrences of each diagnosis for target vs non-target patients
    target_group = all_diagnoses_except_target[all_diagnoses_except_target['subject_id'].isin(target_patients_set)]
    non_target_group = all_diagnoses_except_target[~all_diagnoses_except_target['subject_id'].isin(target_patients_set)]
    
    # Count unique patients with each diagnosis in each group
    target_comorbidities = target_group.groupby('icd_code')['subject_id'].nunique()
    non_target_comorbidities = non_target_group.groupby('icd_code')['subject_id'].nunique()
    
    # Get total patients in each group
    num_target_patients = len(target_patients)
    all_patients = diagnoses_df['subject_id'].unique()
    num_non_target_patients = len(all_patients) - num_target_patients
    
    # Get all unique diagnoses other than target
    all_diagnoses = diagnoses_df[diagnoses_df['icd_code'] != target_icd_code]['icd_code'].unique()
    
    # Create a DataFrame for chi-square with one row per diagnosis
    diagnosis_stats = pd.DataFrame(index=all_diagnoses)
    
    # Get counts of patients in each group with each diagnosis for calculating prevalence
    diagnosis_stats['target_count'] = pd.Series(target_comorbidities)
    diagnosis_stats['non_target_count'] = pd.Series(non_target_comorbidities)
    
    # Fill NaN values with 0
    diagnosis_stats = diagnosis_stats.fillna(0)
    
    # Calculate prevalence percentages
    diagnosis_stats['target_prevalence'] = diagnosis_stats['target_count'] / num_target_patients
    diagnosis_stats['non_target_prevalence'] = diagnosis_stats['non_target_count'] / num_non_target_patients
    
    # Calculate prevalence ratio (avoid division by zero)
    diagnosis_stats['prevalence_ratio'] = np.where(
        diagnosis_stats['non_target_prevalence'] > 0,
        diagnosis_stats['target_prevalence'] / diagnosis_stats['non_target_prevalence'],
        diagnosis_stats['target_prevalence'] / 0.001
    )
    
    # Create contingency tables for chi-square test
    contingency_tables = []
    
    # For each diagnosis, create a 2x2 contingency table
    for diagnosis in diagnosis_stats.index:
        # Patients with this diagnosis and target condition
        with_diag_with_target = diagnosis_stats.loc[diagnosis, 'target_count']
        
        # Patients with this diagnosis but without target condition
        with_diag_without_target = diagnosis_stats.loc[diagnosis, 'non_target_count']
        
        # Patients without this diagnosis but with target condition
        without_diag_with_target = num_target_patients - with_diag_with_target
        
        # Patients without this diagnosis and without target condition
        without_diag_without_target = num_non_target_patients - with_diag_without_target
        
        # Create 2x2 contingency table
        table = np.array([
            [with_diag_with_target, with_diag_without_target],
            [without_diag_with_target, without_diag_without_target]
        ])
        
        contingency_tables.append(table)
    
    # Run chi-square test on each contingency table
    from scipy.stats import chi2_contingency
    
    chi2_values = []
    p_values = []
    
    for table in contingency_tables:
        chi2, p, _, _ = chi2_contingency(table, correction=False)
        chi2_values.append(chi2)
        p_values.append(p)
    
    # Add chi-square results to diagnosis_stats
    diagnosis_stats['chi2_value'] = chi2_values
    diagnosis_stats['p_value'] = p_values
    
    # Reset index to convert to regular DataFrame with 'icd_code' column
    feature_importance = diagnosis_stats.reset_index().rename(columns={'index': 'icd_code'})
    
    # Sort by chi-square value (higher = more predictive)
    feature_importance = feature_importance.sort_values('chi2_value', ascending=False)
    
    # Get top 10 most predictive comorbidities
    print("\nTop 10 Most Significant Comorbidities:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"ICD Code: {row['icd_code']}, Chi2: {row['chi2_value']:.2f}, p-value: {row['p_value']:.5f}")
    
    # Show prevalence comparison for top comorbidities
    print("\nPrevalence Analysis for Top Comorbidities:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"ICD: {row['icd_code']}, Target: {row['target_prevalence']*100:.2f}%, " +
              f"Non-Target: {row['non_target_prevalence']*100:.2f}%, Ratio: {row['prevalence_ratio']:.2f}x")
    
    return feature_importance

def evaluate_threshold_curve(diagnoses_df, target_icd_code, k=10, 
                           sample_size=2000, class_weight=3.0):
    """
    Evaluates model performance across different decision thresholds
    to help find the optimal threshold for minimizing false negatives.
    """
    print(f"Evaluating threshold curve for {target_icd_code}...")
    
    # Build the model
    G, target_patients, non_target_patients = build_balanced_predictor(
        diagnoses_df, target_icd_code, 
        sample_size=sample_size,
        class_weight=class_weight
    )
    
    # Prepare data
    all_patients = target_patients + non_target_patients
    all_labels = [1] * len(target_patients) + [0] * len(non_target_patients)
    
    # Make predictions
    all_predictions = []
    for patient_id in all_patients:
        likelihood = predict_likelihood(G, patient_id, k)
        all_predictions.append(likelihood)
    
    # Evaluate across thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for threshold in thresholds:
        binary_preds = [1 if p >= threshold else 0 for p in all_predictions]
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn)  # Recall
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'false_negatives': fn,
            'false_positives': fp
        })
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\nPerformance across decision thresholds:")
    # Set display options to show the entire table
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None,
                           'display.width', None,
                           'display.precision', 3):
        print(results_df)
    
    # Plot metrics vs threshold
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(results_df['threshold'], results_df['sensitivity'], 'b-', label='Sensitivity/Recall')
    plt.plot(results_df['threshold'], results_df['specificity'], 'r-', label='Specificity')
    plt.plot(results_df['threshold'], results_df['precision'], 'g-', label='Precision')
    plt.plot(results_df['threshold'], results_df['f1'], 'm-', label='F1 Score')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(results_df['threshold'], results_df['false_negatives'], 'r-', label='False Negatives')
    plt.plot(results_df['threshold'], results_df['false_positives'], 'b-', label='False Positives')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Count')
    plt.title('Errors vs. Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'threshold_curve_{target_icd_code}.png')
    plt.show()
    
    # Find threshold that minimizes false negatives while keeping acceptable precision
    best_threshold = results_df[results_df['precision'] >= 0.3].sort_values('false_negatives').iloc[0]
    print(f"\nRecommended threshold to minimize false negatives: {best_threshold['threshold']}")
    print(f"At this threshold: Sensitivity: {best_threshold['sensitivity']:.2f}, Precision: {best_threshold['precision']:.2f}")
    print(f"False Negatives: {best_threshold['false_negatives']}, False Positives: {best_threshold['false_positives']}")
    
    return results_df

# Main code block
if __name__ == "__main__":
    diagnoses_path = "Data/diagnoses_icd.csv"
    
    if os.path.exists(diagnoses_path):
        diagnoses_df = pd.read_csv(diagnoses_path)
    else:
        raise FileNotFoundError(f"Diagnoses file not found at {diagnoses_path}")
    
    # Example: Evaluate prediction model for hypertension (code I10)
    diagnosis_evaluation(
        diagnoses_df, 
        "I10",                  # Hypertension
        k=10,                   # Number of neighbors
        n_folds=5,              # Cross-validation folds
        similarity_threshold=0.1,  # Minimum similarity for connections
        sample_size=2000,       # Sample size for manageable computation
        parallel_jobs=-1
    )
    
    # Find the optimal threshold
    evaluate_threshold_curve(diagnoses_df, "I10")