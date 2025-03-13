import os
import pandas as pd

# Path to the directory containing the hosp dataset CSV files
dataset_path = "Data"

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
    # Save back to the original file
    patients_df.to_csv(patients_file, index=False)
    print(f"Modified patients.csv to keep only subject_id and gender. New shape: {patients_df.shape}")
    
    
# Load admissions.csv and keep only required columns
admissions_file = os.path.join(dataset_path, 'admissions.csv')
if os.path.exists(admissions_file):
    admissions_df = pd.read_csv(admissions_file, low_memory=False)
    # Keep only required columns
    admissions_df = admissions_df[['subject_id', 'insurance', 'language', 'marital_status', 'race']]
    # Save back to the original file
    admissions_df.to_csv(admissions_file, index=False)
    print(f"Modified admissions.csv to keep only required columns. New shape: {admissions_df.shape}")
    
# Merge patients and admissions

# Merge with admissions
merged_df = pd.merge(patients_df, admissions_df, on='subject_id', how='inner')

# Save the merged DataFrame
merged_file = os.path.join(dataset_path, 'merged_patients_admissions.csv')
merged_df.to_csv(merged_file, index=False)
print(f"Merged patients and admissions. Saved to {merged_file}. New shape: {merged_df.shape}")