# Step 2 of the data processing pipeline

# The script reads the diagnoses_icd.csv file, extracts ICD-9 and ICD-10 mental health codes, and exports the first 3 digits of each code as categories to text files for further analysis. This step is useful for simplifying the codes and analyzing trends in mental health diagnoses.


import pandas as pd

# Load the diagnoses_icd file
# Replace 'path_to_file' with the actual path to your diagnoses_icd file
diagnoses_icd = pd.read_csv('Data/diagnoses_icd.csv')

# Replace the icd_code with the first 3 characters of the code. This is the category we will use for analysis.
diagnoses_icd['icd_code'] = diagnoses_icd['icd_code'].str[:3]

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
# Pattern to match codes starting with 290 through 319
pattern = r'^(29[0-9]|30[0-9]|31[0-9]).*'

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

# Export icd_9 and icd_10 categories as txt files for further analysis

# Write ICD-9 categories to a text file
with open('Data/icd_9_categories.txt', 'w') as f:
    for category in icd_9_codes:
        f.write(f"{category}\n")

# Write ICD-10 categories to a text file
with open('Data/icd_10_categories.txt', 'w') as f:
    for category in icd_10_codes:
        f.write(f"{category}\n")
        
# Print a message to confirm the export
print("ICD-9 and ICD-10 categories exported successfully.")