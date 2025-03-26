'''
This script performs exploratory data analysis on the diagnoses_icd table.
It reads the diagnoses_icd.csv file and analyzes the data to understand the distribution of diagnoses per patient.
The analysis includes calculating the number of diagnoses per patient, categorizing patients by the number of diagnoses, and visualizing the distribution.
This step helps identify patterns and trends in the data, such as outliers or data entry errors.
'''

'''
Unfiltered Verification:
Total diagnoses (300606) / Unique patients (10842) = 27.73
Number of patients by diagnosis count:
1-5      2189
6-10     2274
11-15    1545
16-20    1048
21-30    1219
31-50    1159
51-75     588
76+       820

Filtered Verification:
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filter import filter_excessive_diagnoses

diagnoses_df = pd.read_csv('Data/diagnoses_icd.csv')
diagnoses_df = filter_excessive_diagnoses(diagnoses_df, max_diagnoses=100)
# Write filtered data to the same file
diagnoses_df.to_csv('Data/diagnoses_icd.csv', index=False)

# Verification
total_diagnoses = len(diagnoses_df)
unique_patients = diagnoses_df['subject_id'].nunique()
verification = total_diagnoses / unique_patients
print(f"Verification - Total diagnoses ({total_diagnoses}) / Unique patients ({unique_patients}) = {verification:.2f}")

# Compute number of diagnoses per patient
diagnoses_per_patient = diagnoses_df.groupby('subject_id').size()
# Create bins for the number of diagnoses
bins = [0, 5, 10, 15, 20, 30, 50, 75, diagnoses_per_patient.max()]
labels = ['1-5', '6-10', '11-15', '16-20', '21-30', '31-50', '51-75', '76+']

# Categorize patients by number of diagnoses
diagnoses_buckets = pd.cut(diagnoses_per_patient, bins=bins, labels=labels, right=True)
bucket_counts = diagnoses_buckets.value_counts().sort_index()

# Create visualization
plt.figure(figsize=(12, 6))
ax = bucket_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Diagnoses per Patient')
plt.xlabel('Number of Diagnoses')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for i, v in enumerate(bucket_counts):
    ax.text(i, v + 0.5, f"{v:,}", ha='center', va='bottom')

# Add summary statistics
plt.figtext(0.15, 0.85, f"Mean diagnoses per patient: {diagnoses_per_patient.mean():.2f}")
plt.figtext(0.15, 0.82, f"Median diagnoses per patient: {diagnoses_per_patient.median():.0f}")
plt.figtext(0.15, 0.79, f"Max diagnoses per patient: {diagnoses_per_patient.max():.0f}")

plt.tight_layout()
plt.show()

# Show detailed numeric breakdown
print("Number of patients by diagnosis count:")
print(bucket_counts)