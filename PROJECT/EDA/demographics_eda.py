# Step 3 of the data processing pipeline
# The script reads the demographics.csv file and performs exploratory data analysis (EDA) to understand the data better. It includes summary statistics, missing values analysis, unique values in categorical columns, and analysis of demographics and diagnoses data.
# This step helps identify patterns, trends, and potential issues in the data.

import pandas as pd
import os

# Check if the files exist
demographics_path = "Data/demographics.csv"
diagnoses_path = "Data/diagnoses_icd.csv"

# Load the data
if os.path.exists(demographics_path):
    demographics_df = pd.read_csv(demographics_path)
else: 
    raise FileNotFoundError(f"Demographics file not found at {demographics_path}. Try running the demographics.py script first.")

if os.path.exists(diagnoses_path):
    diagnoses_df = pd.read_csv(diagnoses_path)
else: 
    raise FileNotFoundError(f"Diagnoses file not found at {diagnoses_path}. Try running the icd_eda.py script first.")

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

# ICD version counts
icd_counts = diagnoses_df['icd_version'].value_counts()
print("\nICD Version Counts:")
print(icd_counts)

# Top 10 most common ICD codes
icd_counts = diagnoses_df['icd_code'].value_counts().head(10)
print("\nTop 10 Most Common ICD Codes:")
print(icd_counts)

patients_with_diagnoses = len(diagnoses_df['subject_id'].unique())
print(f"Number of patients: {patients_with_diagnoses}")

# Average number of diagnoses per patient
diagnoses_per_patient = diagnoses_df['subject_id'].value_counts()

average_diagnoses_per_patient = diagnoses_per_patient.mean()
print(f"\nAverage number of diagnoses per patient: {average_diagnoses_per_patient:.2f}")

# Median number of diagnoses per patient
median_diagnoses_per_patient = diagnoses_per_patient.median()
print(f"Median number of diagnoses per patient: {median_diagnoses_per_patient:.2f}")

# Mode number of diagnoses per patient
mode_diagnoses_per_patient = diagnoses_per_patient.mode()[0]
print(f"Mode number of diagnoses per patient: {mode_diagnoses_per_patient:.2f}")

# Generate a histogram of the patient diiagnoses
import matplotlib.pyplot as plt
plt.hist(diagnoses_per_patient, bins=range(1, 20), color='skyblue', edgecolor='black', linewidth=1.2)
plt.xlabel('Number of Diagnoses per Patient')
plt.ylabel('Number of Patients')
plt.title('Distribution of Diagnoses per Patient')
plt.show()  # Display the histogram