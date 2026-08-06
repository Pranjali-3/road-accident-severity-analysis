"""
Remove 4500 Slight Injury rows and replace with synthetic Fatal Injury rows.
This rebalances the dataset for better fatal prediction.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

print("Loading data...")
df = pd.read_csv("Road.csv")
print(f"Original: {len(df)} rows")
print(df['Accident_severity'].value_counts())

# Separate by severity
slight = df[df['Accident_severity'] == 'Slight Injury'].copy()
serious = df[df['Accident_severity'] == 'Serious Injury'].copy()
fatal = df[df['Accident_severity'] == 'Fatal injury'].copy()

print(f"\nBefore: Fatal={len(fatal)}, Serious={len(serious)}, Slight={len(slight)}")

# Remove 4500 Slight rows randomly
n_remove = 4500
slight_reduced = slight.sample(n=len(slight) - n_remove, random_state=42)
print(f"Removed {n_remove} Slight rows -> {len(slight_reduced)} remaining")

# Generate synthetic fatal rows based on real fatal patterns
n_generate = n_remove
print(f"\nGenerating {n_generate} synthetic Fatal rows...")

# Analyze real fatal patterns
print("\nFatal patterns:")
print(f"  Time range: {fatal['Time'].mode().values}")
print(f"  Day distribution: {fatal['Day_of_week'].value_counts(normalize=True).head(3).to_dict()}")

# Generate based on real fatal statistics
synthetic_fatal = []

for i in range(n_generate):
    # Sample from real fatal cases with slight variation
    base = fatal.sample(1).iloc[0].copy()
    
    # Add variation to time
    if pd.notna(base['Time']):
        time_str = str(base['Time'])
        try:
            h, m, s = map(int, time_str.split(':'))
            h = (h + np.random.choice([-1, 0, 1])) % 24
            base['Time'] = f"{h:02d}:{m:02d}:{s:02d}"
        except:
            pass
    
    # Slight variation in numeric columns
    if 'Number_of_casualties' in base.index:
        base['Number_of_casualties'] = max(1, base['Number_of_casualties'] + np.random.choice([-1, 0, 0, 1]))
    if 'Number_of_vehicles_involved' in base.index:
        base['Number_of_vehicles_involved'] = max(1, base['Number_of_vehicles_involved'] + np.random.choice([-1, 0, 0, 1]))
    
    synthetic_fatal.append(base)

synthetic_df = pd.DataFrame(synthetic_fatal)
print(f"Generated {len(synthetic_df)} synthetic Fatal rows")

# Combine
new_df = pd.concat([slight_reduced, serious, fatal, synthetic_df], ignore_index=True)
new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nNew dataset: {len(new_df)} rows")
print(new_df['Accident_severity'].value_counts())

# Save
new_df.to_csv("Road_balanced.csv", index=False)
print(f"\nSaved: Road_balanced.csv")
