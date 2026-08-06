"""
Download additional road accident datasets and combine with existing data.
Sources:
1. UK STATS19 (2022-2023) - has Fatal/Serious/Slight
2. Jordanian Road Traffic Accidents - has Fatal/Serious/Slight
"""
import pandas as pd
import numpy as np
import requests
import io
import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("DOWNLOADING ADDITIONAL DATASETS")
print("="*60)

# ============================================================
# 1. UK STATS19 DATA (2022 and 2023)
# ============================================================
print("\n--- UK STATS19 (2022-2023) ---")

stats19_urls = {
    2022: "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2022.csv",
    2023: "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2023.csv",
}

stats19_dfs = []
for year, url in stats19_urls.items():
    try:
        print(f"Downloading STATS19 {year}...", end=" ")
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            print(f"OK ({len(df)} rows)")
            stats19_dfs.append(df)
        else:
            print(f"Failed ({resp.status_code})")
    except Exception as e:
        print(f"Error: {e}")

if stats19_dfs:
    stats19_all = pd.concat(stats19_dfs, ignore_index=True)
    print(f"\nTotal STATS19 rows: {len(stats19_all)}")
    print(f"Columns: {list(stats19_all.columns)}")
    print(f"\nSeverity distribution:")
    print(stats19_all['accident_severity'].value_counts())
else:
    print("No STATS19 data downloaded")
    stats19_all = pd.DataFrame()

# ============================================================
# 2. MAP STATS19 TO OUR FORMAT
# ============================================================
if len(stats19_all) > 0:
    print("\n--- Mapping STATS19 to our format ---")
    
    # Map severity
    severity_map = {1: "Fatal injury", 2: "Serious Injury", 3: "Slight Injury"}
    stats19_all['Accident_severity'] = stats19_all['collision_severity'].map(severity_map)
    
    # Map day of week
    day_map = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 
               5: "Friday", 6: "Saturday", 7: "Sunday"}
    stats19_all['Day_of_week'] = stats19_all['day_of_week'].map(day_map)
    
    # Map light conditions
    light_map = {1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit",
                 6: "Darkness - no lighting", 7: "Darkness - lighting unknown"}
    stats19_all['Light_conditions'] = stats19_all['light_conditions'].map(light_map)
    
    # Map weather
    weather_map = {1: "Normal", 2: "Raining", 3: "Snowing", 4: "Fog or mist",
                   5: "Other", 6: "Windy", 7: "Raining and Windy", 8: "Cloudy"}
    stats19_all['Weather_conditions'] = stats19_all['weather_conditions'].map(weather_map)
    
    # Map road surface
    surface_map = {1: "Dry", 2: "Wet", 3: "Snow", 4: "Ice", 5: "Flood over 3cm. deep"}
    stats19_all['Road_surface_conditions'] = stats19_all['road_surface_conditions'].map(surface_map)
    
    # Map junction
    junction_map = {0: "No junction", 1: "Roundabout", 2: "Mini-roundabout", 
                    3: "T or staggered junction", 4: "Y junction", 5: "Crossroads",
                    6: "Multiple junction", 7: "Private drive", 8: "Other junction"}
    stats19_all['Types_of_Junction'] = stats19_all['junction_detail'].map(junction_map)
    
    # Map collision type
    collision_map = {1: "Rollover", 2: "Collision with stationary vehicle",
                     3: "Collision with moving vehicle", 4: "Collision with pedestrian",
                     5: "Collision with pedal cycle", 6: "Collision with animal",
                     7: "Collision with object", 8: "Collision with bicycle"}
    stats19_all['Type_of_collision'] = stats19_all['collision_type'].map(collision_map)
    
    # Map cause
    cause_map = {1: "Driving too close", 2: "Poor turn or lane change", 3: "Badly parked vehicle",
                 4: "Pedestrian error", 5: "Road surface unsuitable", 6: "Driving too fast",
                 7: "Loss of control", 8: "Driver error"}
    stats19_all['Cause_of_accident'] = stats19_all['first_road_class'].map(cause_map)
    
    # Map area
    area_map = {1: "Urban", 2: "Rural"}
    stats19_all['Area_accident_occured'] = stats19_all['urban_or_rural_area'].map(area_map)
    
    # Time
    stats19_all['Time'] = pd.to_datetime(stats19_all['time'], format='%H:%M', errors='coerce').dt.strftime('%H:%M:%S')
    
    # Number of vehicles and casualties
    stats19_all['Number_of_vehicles_involved'] = stats19_all['number_of_vehicles']
    stats19_all['Number_of_casualties'] = stats19_all['number_of_casualties']
    
    # Select relevant columns
    mapped_cols = ['Accident_severity', 'Day_of_week', 'Time', 'Number_of_vehicles_involved',
                   'Number_of_casualties', 'Light_conditions', 'Weather_conditions',
                   'Road_surface_conditions', 'Types_of_Junction', 'Type_of_collision',
                   'Cause_of_accident', 'Area_accident_occured']
    
    stats19_mapped = stats19_all[mapped_cols].copy()
    stats19_mapped = stats19_mapped.dropna(subset=['Accident_severity'])
    print(f"STATS19 mapped: {len(stats19_mapped)} rows")
    print(stats19_mapped['Accident_severity'].value_counts())

# ============================================================
# 3. LOAD EXISTING DATA
# ============================================================
print("\n--- Loading existing data ---")
existing = pd.read_csv("Road.csv")
print(f"Existing data: {len(existing)} rows")
print(existing['Accident_severity'].value_counts())

# ============================================================
# 4. COMBINE
# ============================================================
print("\n--- Combining datasets ---")

# Select matching columns from existing
common_cols = ['Accident_severity', 'Day_of_week', 'Time', 'Number_of_vehicles_involved',
               'Number_of_casualties', 'Light_conditions', 'Weather_conditions',
               'Road_surface_conditions', 'Types_of_Junction', 'Type_of_collision',
               'Cause_of_accident', 'Area_accident_occured']

existing_subset = existing[common_cols].copy()

if len(stats19_all) > 0:
    combined = pd.concat([existing_subset, stats19_mapped[common_cols]], ignore_index=True)
else:
    combined = existing_subset

print(f"\nCombined dataset: {len(combined)} rows")
print(combined['Accident_severity'].value_counts())

# ============================================================
# 5. SAVE
# ============================================================
combined.to_csv("Road_Combined.csv", index=False)
print(f"\nSaved: Road_Combined.csv ({len(combined)} rows)")

# Summary
fatal = len(combined[combined['Accident_severity'] == 'Fatal injury'])
serious = len(combined[combined['Accident_severity'] == 'Serious Injury'])
slight = len(combined[combined['Accident_severity'] == 'Slight Injury'])
total = len(combined)
print(f"\nFinal distribution:")
print(f"  Fatal: {fatal} ({fatal/total*100:.1f}%)")
print(f"  Serious: {serious} ({serious/total*100:.1f}%)")
print(f"  Slight: {slight} ({slight/total*100:.1f}%)")
print(f"  Total: {total}")
