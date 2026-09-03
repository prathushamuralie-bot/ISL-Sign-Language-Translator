"""
Sign Language Recognition - Remove Wrong Label Samples
============================================================
Idhu script, dataset la irukura oru specific label oda ELLAA samples ah remove pannum.
Use case: thappa vera sign ku samples add pannitinga na, andha ellame delete panni,
puthusa correct ah re-record pannalam.
"""

import pandas as pd

CSV_FILE = "dataset/sign_data_two_hands.csv"   # unga current dataset file (single hand na "sign_data.csv")
LABEL_TO_REMOVE = "name"                # <-- edha label ku wrong samples irukko, idha podunga

df = pd.read_csv(CSV_FILE)

print("BEFORE removal:")
print(df['label'].value_counts())

before_count = len(df)
df = df[df['label'] != LABEL_TO_REMOVE]
after_count = len(df)

print(f"\nRemoved {before_count - after_count} samples with label '{LABEL_TO_REMOVE}'")

df.to_csv(CSV_FILE, index=False)

print("\nAFTER removal:")
print(df['label'].value_counts())
print(f"\nSaved: {CSV_FILE}")
print(f"Ippo '{LABEL_TO_REMOVE}' ku fresh ah correct samples record pannunga (data_collection tool use panni).")
