import pandas as pd

# Load the diagnoses_icd file
# Replace 'path_to_file' with the actual path to your diagnoses_icd file
diagnoses_icd = pd.read_csv('Data/diagnoses_icd.csv')

# Display the first few rows of the dataframe
diagnoses_icd.head()

# Count the occurrences of each ICD version
icd_counts = diagnoses_icd['icd_version'].value_counts()

# Display the counts
print(icd_counts)

# ICD 10 mental health codes start with F
icd_10_codes = set(diagnoses_icd.loc[(diagnoses_icd['icd_code'].str.startswith('F', na=False)) & (diagnoses_icd['icd_version'] == 10), 'icd_code'])
print(f"Found {len(icd_10_codes)} unique ICD codes starting with F in ICD-10:")
print(icd_10_codes)

# ICD 9 mental health codes start with numbers 290-319
# Get all ICD-9 codes that start with numbers 290-319

# Pattern to match codes starting with 290 through 319, possibly followed by other digits
pattern = r'^(29[0-9]|30[0-9]|31[0-9])'

# Filter for ICD-9 codes matching our pattern
icd_9_mental = diagnoses_icd.loc[(diagnoses_icd['icd_code'].str.match(pattern, na=False)) & (diagnoses_icd['icd_version'] == 9), 'icd_code']

# Get unique codes
icd_9_codes = set(icd_9_mental)

print(f"\nFound {len(icd_9_codes)} unique ICD mental health codes (290-319) in ICD-9:")
print(sorted(list(icd_9_codes))[:20], "...")  # Show first 20 codes

# Count occurrences of each code
top_codes = icd_9_mental.value_counts().head(10)
print("\nTop 10 most common mental health ICD-9 codes:")
print(top_codes)