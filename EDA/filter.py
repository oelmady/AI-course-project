'''
Filtering out 524 patients with more than 100 diagnoses
These patients represent 4.83% of total patients
Removing 99912 records (33.24% of all records)
Verification - Total diagnoses (200694) / Unique patients (10318) = 19.45
Number of patients by diagnosis count:
1-5      2189
6-10     2274
11-15    1545
16-20    1048
21-30    1219
31-50    1159
51-75     588
76+       296
'''
def filter_excessive_diagnoses(diagnoses_df, max_diagnoses=100):
    """
    Filters out patients with excessive number of diagnoses (likely data entry errors).
    
    Args:
        diagnoses_df: DataFrame with diagnoses data
        max_diagnoses: Maximum number of diagnoses allowed per patient
        
    Returns:
        DataFrame: Filtered diagnoses DataFrame
    """
    # Count diagnoses per patient
    diagnoses_per_patient = diagnoses_df.groupby('subject_id').size()
    
    # Identify patients with excessive diagnoses
    excessive_patients = diagnoses_per_patient[diagnoses_per_patient > max_diagnoses].index
    
    # Report on filtering
    print(f"Filtering out {len(excessive_patients)} patients with more than {max_diagnoses} diagnoses")
    print(f"These patients represent {len(excessive_patients) / diagnoses_df['subject_id'].nunique() * 100:.2f}% of total patients")
    
    # Get total diagnoses for these patients
    excessive_records = diagnoses_df[diagnoses_df['subject_id'].isin(excessive_patients)]
    print(f"Removing {len(excessive_records)} records ({len(excessive_records) / len(diagnoses_df) * 100:.2f}% of all records)")
    
    # Filter out these patients
    filtered_df = diagnoses_df[~diagnoses_df['subject_id'].isin(excessive_patients)]
    
    return filtered_df
