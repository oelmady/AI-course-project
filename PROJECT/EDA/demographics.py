# Step 1 of the data processing pipeline
# This script unzips the CSV files in the Data folder and merges the patients and admissions data into a single CSV file, demographics.csv
import os
import pandas as pd

# Path to the directory containing the hosp dataset CSV files
dataset_path = "../Data"
demographics_path = 'demographics.csv'

# Only perform this processing step if we do not have demographics.csv
if os.path.exists(os.path.join(dataset_path, demographics_path)): 
    print("Demographics.csv already exists. Exiting...")
    exit(1)

# Unzip all files in the Data folder and export them as CSV files
for file in os.listdir(dataset_path):
    if file.endswith('.gz'):
        file_path = os.path.join(dataset_path, file)
        output_file = os.path.join(dataset_path, file.replace('.gz', ''))
        
        # Read the gzipped file
        print(f"Extracting {file}...")
        df_temp = pd.read_csv(file_path, compression='gzip', low_memory=False)
        
        # Save as regular CSV
        df_temp.to_csv(output_file, index=False)
        print(f"Saved as {output_file}")

patients_file = os.path.join(dataset_path, 'patients.csv')
if os.path.exists(patients_file):
    patients_df = pd.read_csv(patients_file, low_memory=False)
    # Keep only required columns
    patients_df = patients_df[['subject_id', 'gender']]
    
# Load admissions.csv and keep only required columns
admissions_file = os.path.join(dataset_path, 'admissions.csv')
if os.path.exists(admissions_file):
    admissions_df = pd.read_csv(admissions_file, low_memory=False)
    # Keep only required columns
    admissions_df = admissions_df[['subject_id', 'insurance', 'language', 'marital_status', 'race']]
   
# Merge patients and admissions
merged_df = pd.merge(patients_df, admissions_df, on='subject_id', how='inner')

# Drop duplicate rows based on subject_id
merged_df = merged_df.drop_duplicates(subset='subject_id')

# Process race column to remove detailed classifications after " - "
merged_df['race'] = merged_df['race'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) and ' - ' in x else x)

# Save the merged DataFrame
merged_file = os.path.join(dataset_path, demographics_path)
merged_df.to_csv(merged_file, index=False)
print(f"Merged patients and admissions. Saved to {merged_file}. New shape: {merged_df.shape}")
