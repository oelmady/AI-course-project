import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

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
    
    # Sample patients if needed
    if sample_size and len(patient_diagnoses) > sample_size:
        from modules.data_loader import sample_patients
        target_patients, non_target_patients = sample_patients(
            target_patients, non_target_patients, sample_size)
        
        sample_patients_set = target_patients.union(non_target_patients)
        patient_diagnoses = patient_diagnoses[patient_diagnoses['subject_id'].isin(sample_patients_set)]
        
        print(f"Using a representative sample of {len(patient_diagnoses)} patients")
    
    # Step 2: Process demographics
    patient_ids = patient_diagnoses['subject_id'].tolist()
    demographics_map = {}
    cols_to_encode = []
    
    if include_demographics and demographics_df is not None:
        demographics_map, patient_ids, target_patients, non_target_patients, cols_to_encode = process_demographics(
            demographics_df, patient_ids, target_patients, non_target_patients)
        # Update patient_diagnoses to only include patients with demographic data
        patient_diagnoses = patient_diagnoses[patient_diagnoses['subject_id'].isin(patient_ids)]
    
    # Step 3: Convert diagnoses to binary indicators and calculate similarities
    X, mlb = create_diagnosis_features(
        patient_diagnoses, target_icd_code, use_matrix_format)
    
    # Step 4: Calculate patient similarities
    print("Computing patient similarities...")
    similarity_matrix = cosine_similarity(X, dense_output=False if use_matrix_format else True)
    
    # Step 5: Create a graph representation
    G = create_patient_network(
        similarity_matrix, patient_ids, target_patients, similarity_threshold,
        include_demographics, demographics_map, cols_to_encode, class_weight,
        demographic_weight, use_matrix_format)
    
    return G, list(target_patients), list(non_target_patients)

def process_demographics(demographics_df, patient_ids, target_patients, non_target_patients):
    """Process demographic features for patients"""
    print("Processing demographic features...")
    # Get demographics for patients in our dataset
    demo_features = demographics_df[demographics_df['subject_id'].isin(patient_ids)].copy()
    
    # Ensure all patients have demographic records
    patient_with_demos = set(demo_features['subject_id'])
    patients_without_demos = set(patient_ids) - patient_with_demos
    
    if patients_without_demos:
        print(f"Warning: {len(patients_without_demos)} patients don't have demographic data")
        # Keep only patients with demographic data
        patient_ids = [pid for pid in patient_ids if pid in patient_with_demos]
        target_patients = target_patients.intersection(patient_with_demos)
        non_target_patients = non_target_patients.intersection(patient_with_demos)
    
    # One-hot encode categorical variables
    categorical_cols = ['gender', 'insurance', 'race', 'language', 'marital_status']
    # Only include columns that actually exist in the DataFrame
    cols_to_encode = [col for col in categorical_cols if col in demo_features.columns]
    
    demographics_map = {}
    if cols_to_encode:
        demo_encoded = pd.get_dummies(demo_features, columns=cols_to_encode, drop_first=False)
        
        # Create a mapping from patient_id to demographics feature vector
        demographics_map = {row['subject_id']: row.drop('subject_id').values 
                           for _, row in demo_encoded.iterrows()}
    else:
        print("Warning: No categorical demographic features found")
    
    return demographics_map, patient_ids, target_patients, non_target_patients, cols_to_encode

def create_diagnosis_features(patient_diagnoses, target_icd_code, use_matrix_format):
    """Create binary features for patient diagnoses"""
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
        X = diagnoses_matrix.copy()
        if target_feature_idx >= 0:
            X = X.tolil()
            X[:, target_feature_idx] = 0
            X = X.tocsr()
    else:
        if target_feature_idx >= 0:
            mask = np.ones(len(mlb.classes_), dtype=bool)
            mask[target_feature_idx] = False
            X = diagnoses_matrix[:, mask]
        else:
            X = diagnoses_matrix
            
    return X, mlb

def create_patient_network(similarity_matrix, patient_ids, target_patients, 
                          similarity_threshold, include_demographics, demographics_map, 
                          cols_to_encode, class_weight, demographic_weight, use_matrix_format):
    """Create a network graph of patients connected by similarities"""
    print("Building patient similarity network...")
    G = nx.Graph()
    
    # Add nodes (patients)
    for i, patient_id in enumerate(patient_ids):
        has_target = patient_id in target_patients
        node_weight = class_weight if has_target else 1.0
        
        # Add demographics as node attributes if available
        if include_demographics and demographics_map and cols_to_encode:
            G.add_node(patient_id, has_target=has_target, weight=node_weight, 
                      demographics=demographics_map.get(patient_id, None))
        else:
            G.add_node(patient_id, has_target=has_target, weight=node_weight)
    
    # Add edges (significant similarities between patients)
    edge_count = 0
    
    if use_matrix_format:
        # Process sparse similarity matrix
        edge_count = add_edges_sparse(G, similarity_matrix, patient_ids, 
                                    similarity_threshold, include_demographics, 
                                    demographics_map, cols_to_encode, demographic_weight)
    else:
        # Process dense similarity matrix
        edge_count = add_edges_dense(G, similarity_matrix, patient_ids, 
                                   similarity_threshold, include_demographics, 
                                   demographics_map, cols_to_encode, demographic_weight)
    
    print(f"Network built with {G.number_of_nodes()} patients and {edge_count} connections")
    return G

def add_edges_sparse(G, similarity_matrix, patient_ids, similarity_threshold,
                    include_demographics, demographics_map, cols_to_encode, demographic_weight):
    """Add edges to graph from sparse similarity matrix"""
    edge_count = 0
    cx = similarity_matrix.tocoo()
    for i, j, sim in zip(cx.row, cx.col, cx.data):
        if i < j and sim > similarity_threshold:  # Avoid duplicates and self-loops
            # If demographics are included, adjust similarity by demographic similarity
            if include_demographics and demographics_map and cols_to_encode:
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
    return edge_count

def add_edges_dense(G, similarity_matrix, patient_ids, similarity_threshold,
                   include_demographics, demographics_map, cols_to_encode, demographic_weight):
    """Add edges to graph from dense similarity matrix"""
    edge_count = 0
    rows, cols = np.where(similarity_matrix > similarity_threshold)
    for i, j in zip(rows, cols):
        if i < j:  # Avoid duplicates and self-loops
            # Similar demographic adjustment as above
            if include_demographics and demographics_map and cols_to_encode:
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
    return edge_count

def predict_likelihood(G, patient_id, k=10):
    """
    Predicts likelihood of a patient having the target diagnosis based on k nearest neighbors.
    
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
    """Process a batch of patients in parallel for prediction"""
    results = []
    for patient_id in patient_batch:
        likelihood = predict_likelihood(G, patient_id, k)
        results.append((patient_id, likelihood))
    return results