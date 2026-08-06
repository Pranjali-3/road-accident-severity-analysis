"""
Better balanced dataset: boost Serious class too.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv("Road.csv")
print(f"Original: {len(df)} rows")
print(df['Accident_severity'].value_counts())

# Separate
slight = df[df['Accident_severity'] == 'Slight Injury'].copy()
serious = df[df['Accident_severity'] == 'Serious Injury'].copy()
fatal = df[df['Accident_severity'] == 'Fatal injury'].copy()

print(f"\nOriginal: Fatal={len(fatal)}, Serious={len(serious)}, Slight={len(slight)}")

# Target: ~5000 Fatal, ~4000 Serious, ~5000 Slight
target_fatal = 5000
target_serious = 4000
target_slight = 5000

# Remove Slight
n_remove_slight = len(slight) - target_slight
slight_reduced = slight.sample(n=len(slight) - n_remove_slight, random_state=42)
print(f"Removed {n_remove_slight} Slight rows")

# Generate more Fatal
n_gen_fatal = target_fatal - len(fatal)
print(f"Generating {n_gen_fatal} Fatal rows...")
synthetic_fatal = []
for i in range(n_gen_fatal):
    base = fatal.sample(1).iloc[0].copy()
    if pd.notna(base['Time']):
        try:
            h, m, s = map(int, str(base['Time']).split(':'))
            h = (h + np.random.choice([-2,-1,0,1,2])) % 24
            base['Time'] = f"{h:02d}:{m:02d}:{s:02d}"
        except: pass
    base['Number_of_casualties'] = max(1, base['Number_of_casualties'] + np.random.choice([-1,0,0,1]))
    base['Number_of_vehicles_involved'] = max(1, base['Number_of_vehicles_involved'] + np.random.choice([-1,0,0,1]))
    synthetic_fatal.append(base)

# Generate more Serious
n_gen_serious = target_serious - len(serious)
print(f"Generating {n_gen_serious} Serious rows...")
synthetic_serious = []
for i in range(n_gen_serious):
    base = serious.sample(1).iloc[0].copy()
    if pd.notna(base['Time']):
        try:
            h, m, s = map(int, str(base['Time']).split(':'))
            h = (h + np.random.choice([-2,-1,0,1,2])) % 24
            base['Time'] = f"{h:02d}:{m:02d}:{s:02d}"
        except: pass
    base['Number_of_casualties'] = max(1, base['Number_of_casualties'] + np.random.choice([-1,0,0,1]))
    base['Number_of_vehicles_involved'] = max(1, base['Number_of_vehicles_involved'] + np.random.choice([-1,0,0,1]))
    synthetic_serious.append(base)

# Combine
fatal_all = pd.concat([fatal, pd.DataFrame(synthetic_fatal)])
serious_all = pd.concat([serious, pd.DataFrame(synthetic_serious)])

combined = pd.concat([slight_reduced, serious_all, fatal_all], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nNew dataset: {len(combined)} rows")
print(combined['Accident_severity'].value_counts())

combined.to_csv("Road_balanced_v2.csv", index=False)
print("Saved: Road_balanced_v2.csv")
