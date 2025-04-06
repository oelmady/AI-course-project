import os
import pandas as pd

diagnoses_path = "../Data/diagnoses_icd.csv"

def analyze_comorbidities_and_demographics(icd_category):
    """
    Analyzes comorbidities and demographics for patients with a specific ICD Category.
    
    Args:
        icd_code (str): The ICD Category to analyze
        
    Returns:
        None: Prints analysis results
    """
    print(f"\n\n--- Analysis for ICD Category: {icd_category} ---")
    
    # Find all patients with the specified ICD Category (first 3 characters of ICD code)
    patients_with_condition = diagnoses_df[diagnoses_df['icd_code'].str.startswith(icd_category[:3])]['subject_id'].unique()
    num_patients = len(patients_with_condition)
    
    if num_patients == 0:
        print(f"No patients found with ICD Category {icd_category}")
        return
    
    print(f"Number of patients with this diagnosis: {num_patients}")

    # Find comorbid conditions (other diagnoses these patients have)
    # Ensure the diagnosis is not in the same category (does not start with the same prefix)
    comorbid_conditions = diagnoses_df[
        (diagnoses_df['subject_id'].isin(patients_with_condition)) & 
        (~diagnoses_df['icd_code'].str.startswith(icd_category[:3]))
    ]
    
    # Count unique patients with each comorbid condition instead of raw occurrences
    # First get unique patient-diagnosis pairs
    unique_patient_diagnoses = comorbid_conditions[['subject_id', 'icd_code']].drop_duplicates()
    
    # Then count how many unique patients have each diagnosis
    condition_counts = unique_patient_diagnoses['icd_code'].value_counts()
    
    # Calculate percentage of patients with each comorbidity
    condition_percentages = (condition_counts / num_patients) * 100
    
    # Display top 5 comorbid conditions
    print("\nTop 5 Comorbid Conditions:")
    if len(condition_counts) > 0:
        for code, count in condition_counts.head(5).items():
            percentage = condition_percentages[code]
            print(f"ICD Diagnosis: {code}, Count: {count}, Percentage: {percentage:.2f}%")
        
        # Statistics for comorbid conditions per patient
        comorbid_per_patient = comorbid_conditions.groupby('subject_id').size()
        
        print("\nComorbidity Statistics (other conditions per patient):")
        print(f"Mean: {comorbid_per_patient.mean():.2f}")
        print(f"Median: {comorbid_per_patient.median():.2f}")
        print(f"Min: {comorbid_per_patient.min()}")
        print(f"Max: {comorbid_per_patient.max()}")
    else:
        print("No comorbid conditions found")
    
    # Demographic analysis for patients with the condition
    print("\nDemographic Profile for Patients with this Diagnosis:")
    
    demographics_df = pd.read_csv('Data/demographics.csv')
    
    # Get demographics for these patients
    patient_demographics = demographics_df[
        demographics_df['subject_id'].isin(patients_with_condition)]
    
    # Gender distribution
    gender_dist = patient_demographics['gender'].value_counts(normalize=True) * 100
    print("\nGender Distribution:")
    for gender, percentage in gender_dist.items():
        print(f"{gender}: {percentage:.2f}%")
    
    # Race distribution
    race_dist = patient_demographics['race'].value_counts(normalize=True) * 100
    print("\nRace Distribution:")
    for race, percentage in race_dist.head(5).items():
        print(f"{race}: {percentage:.2f}%")
    
    # Insurance distribution
    insurance_dist = patient_demographics['insurance'].value_counts(normalize=True) * 100
    print("\nInsurance Distribution:")
    for insurance, percentage in insurance_dist.items():
        print(f"{insurance}: {percentage:.2f}%")
    
    # Language distribution
    language_dist = patient_demographics['language'].value_counts(normalize=True) * 100
    print("\nTop 5 Languages:")
    for language, percentage in language_dist.head(5).items():
        print(f"{language}: {percentage:.2f}%")

# Example usage
if __name__ == "__main__":
    # Fix the typo in the original code
    if os.path.exists(diagnoses_path):
        diagnoses_df = pd.read_csv(diagnoses_path)
    else:
        raise FileNotFoundError(f"Diagnoses file not found at {diagnoses_path}. Try running the icd_eda.py script first.")
    
    # Example: Analyze comorbidities for a specific ICD Category
    
    analyze_comorbidities_and_demographics("I10")  # Example: Hypertension
    analyze_comorbidities_and_demographics("F30") 