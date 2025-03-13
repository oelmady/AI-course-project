import pandas as pd
import os

dataset_path = "Data"
# Check if the files exist
demographics_path = os.path.join(dataset_path, "demographics.csv")
diagnoses_path = os.path.join(dataset_path, "diagnoses_icd.csv")

# Load the data
if os.path.exists(demographics_path):
    demographics_df = pd.read_csv(demographics_path)
else: 
    raise FileNotFoundError(f"Demographics file not found at {demographics_path}")

diagnoses_df = pd.read_csv(diagnoses_path)

# Summary statistics
print("\nDemographics Summary Statistics:")
demographics_df.describe(include='all')

print("\nDiagnoses Summary Statistics:")
diagnoses_df.describe(include='all')

# Missing or null values 
print("\n--- Missing Values Analysis ---")
missing_values = demographics_df.isnull().sum()
print("\nMissing Values in Demographics DataFrame:")
print(missing_values[missing_values > 0])
missing_values_diagnoses = diagnoses_df.isnull().sum()
print("\nMissing Values in Diagnoses DataFrame:")
print(missing_values_diagnoses[missing_values_diagnoses > 0])

# Check for duplicates
duplicates = demographics_df.duplicated().sum()
print(f"\nNumber of duplicate rows in Demographics DataFrame: {duplicates}")
duplicates_diagnoses = diagnoses_df.duplicated().sum()
print(f"Number of duplicate rows in Diagnoses DataFrame: {duplicates_diagnoses}")

# Check for unique values in categorical columns
print("\n--- Unique Values in Categorical Columns ---")
categorical_columns = demographics_df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    unique_values = demographics_df[col].unique()
    print(f"\nUnique values in {col}:")
    print(unique_values)

# Demographics analysis
print("\n--- Demographics Analysis ---")

# Gender distribution
gender_counts = demographics_df['gender'].value_counts()
print("\nGender Distribution:")
print(gender_counts)

# Insurance distribution
insurance_counts = demographics_df['insurance'].value_counts()
print("\nInsurance Distribution:")
print(insurance_counts)

# Language distribution
language_counts = demographics_df['language'].value_counts().head(10)
print("\nTop 10 Languages:")
print(language_counts)

# Marital status distribution
marital_counts = demographics_df['marital_status'].value_counts()
print("\nMarital Status Distribution:")
print(marital_counts)

# Race distribution
race_counts = demographics_df['race'].value_counts()
print("\nRace Distribution:")
print(race_counts)

# Diagnoses analysis
print("\n--- Diagnoses Analysis ---")

# Count of diagnoses per patient
diagnoses_per_patient = diagnoses_df.groupby('subject_id').size()
print("\nDiagnoses per Patient Statistics:")
print(diagnoses_per_patient.describe())

# Top 10 most common ICD codes
icd_counts = diagnoses_df['icd_code'].value_counts().head(10)
print("\nTop 10 Most Common ICD Codes:")
print(icd_counts)

# Correlation between demographics and diagnoses (if possible)

print("\nPatients with diagnoses recorded:")
patients_with_diagnoses = len(diagnoses_df['subject_id'].unique())
total_patients = len(demographics_df['subject_id'].unique())
print(f"Number of patients with diagnoses: {patients_with_diagnoses}")
print(f"Total number of patients: {total_patients}")
print(f"Percentage: {(patients_with_diagnoses/total_patients)*100:.2f}%")

