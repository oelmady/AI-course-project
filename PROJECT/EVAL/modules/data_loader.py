import os
import pandas as pd

def load_diagnoses_data(dataset_path="../../PROJECT/Data"):
    """Load diagnoses data and return as DataFrame"""
    diagnoses_path = os.path.join(dataset_path, "diagnoses_icd.csv")
    if os.path.exists(diagnoses_path):
        diagnoses_df = pd.read_csv(diagnoses_path)
        print(f"Loaded diagnoses data with {len(diagnoses_df)} records")
        return diagnoses_df
    else:
        raise FileNotFoundError(f"Diagnoses file not found at {diagnoses_path}")

def load_demographics_data(dataset_path="../../PROJECT/Data"):
    """Load demographics data and return as DataFrame"""
    demographics_path = os.path.join(dataset_path, "demographics.csv")
    if os.path.exists(demographics_path):
        df = pd.read_csv(demographics_path)
        print(f"Loaded demographics data with {len(df)} records and {len(df.columns)} features")
        return df
    else:
        raise FileNotFoundError(f"Demographics file not found at {demographics_path}")

def sample_patients(target_patients, non_target_patients, sample_size):
    """
    Create a balanced sample of patients with and without the target diagnosis.
    
    Args:
        target_patients: Set of patient IDs with the target diagnosis
        non_target_patients: Set of patient IDs without the target diagnosis
        sample_size: Desired sample size
        
    Returns:
        Sets of sampled target and non-target patients
    """
    # Calculate sampling proportions
    target_ratio = len(target_patients) / (len(target_patients) + len(non_target_patients))
    target_sample_size = int(sample_size * target_ratio)
    non_target_sample_size = sample_size - target_sample_size
    
    # Sample from each group
    target_sample = list(target_patients)
    non_target_sample = list(non_target_patients)
    
    import numpy as np
    np.random.shuffle(target_sample)
    np.random.shuffle(non_target_sample)
    
    target_sample = target_sample[:target_sample_size]
    non_target_sample = non_target_sample[:non_target_sample_size]
    
    return set(target_sample), set(non_target_sample)