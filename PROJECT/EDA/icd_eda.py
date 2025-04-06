# Step 2 of the data processing pipeline

import pandas as pd

# Load the diagnoses_icd file
diagnoses_icd = pd.read_csv('../Data/diagnoses_icd.csv')

# Keep the category for reference
diagnoses_icd['icd_category'] = diagnoses_icd['icd_code'].str[:3]

# Display the first few rows of the dataframe
diagnoses_icd.head()

# Count the occurrences of each ICD version
icd_counts = diagnoses_icd['icd_version'].value_counts()

# Display the counts
print(icd_counts)

# ICD 10 mental health codes start with F
icd_10_exact = diagnoses_icd.loc[(diagnoses_icd['icd_code'].str.startswith('F', na=False)) & 
                               (diagnoses_icd['icd_version'] == 10), 'icd_code']
icd_10_codes_exact = set(icd_10_exact)
print(f"Found {len(icd_10_codes_exact)} unique ICD-10 mental health diagnoses:")
print(sorted(list(icd_10_codes_exact))[:20], "...")  # Show first 20 codes

# Count and display top 10 most frequent ICD-10 mental health diagnoses
top_icd10_codes = icd_10_exact.value_counts().head(10)
print("\nTop 10 most common mental health ICD-10 diagnoses:")
for code, count in top_icd10_codes.items():
    print(f"  {code}: {count:,} occurrences")

# ICD 9 mental health codes - exact diagnoses between 290-319
pattern = r'^(29[0-9]|30[0-9]|31[0-9]).*'

# Filter for exact ICD-9 codes matching our pattern
icd_9_mental_exact = diagnoses_icd.loc[(diagnoses_icd['icd_code'].str.match(pattern, na=False)) & 
                                     (diagnoses_icd['icd_version'] == 9), 'icd_code']

# Get unique codes
icd_9_codes_exact = set(icd_9_mental_exact)

print(f"\nFound {len(icd_9_codes_exact)} unique ICD-9 mental health diagnoses (290-319):")
print(sorted(list(icd_9_codes_exact))[:20], "...")  # Show first 20 codes

# Count occurrences of each code
top_icd9_codes = icd_9_mental_exact.value_counts().head(10)
print("\nTop 10 most common mental health ICD-9 diagnoses:")
for code, count in top_icd9_codes.items():
    print(f"  {code}: {count:,} occurrences")

# Export exact ICD-9 and ICD-10 diagnoses as txt files for review
with open('Data/icd_9_exact_diagnoses.txt', 'w') as f:
    for diagnosis in sorted(icd_9_codes_exact):
        f.write(f"{diagnosis}\n")

with open('Data/icd_10_exact_diagnoses.txt', 'w') as f:
    for diagnosis in sorted(icd_10_codes_exact):
        f.write(f"{diagnosis}\n")
        
print("Exact ICD-9 and ICD-10 diagnoses exported successfully.")