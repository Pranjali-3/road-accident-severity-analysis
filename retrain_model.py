"""
Retrain model with class weights + feature selection for better fatal recall.
"""
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, recall_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv("Road.csv")
print(f"Original shape: {df.shape}")

# ============================================================
# 2. FEATURE ENGINEERING (same as notebook)
# ============================================================
df_fe = df.copy()

# Drop columns not needed
if "Time" in df_fe.columns:
    df_fe["Hour"] = pd.to_datetime(df_fe["Time"], format="%H:%M:%S", errors="coerce").dt.hour
    df_fe["Time_Period"] = pd.cut(
        df_fe["Hour"],
        bins=[-1, 5, 9, 17, 21, 24],
        labels=["Late Night", "Morning Rush", "Daytime", "Evening Rush", "Night"],
        include_lowest=True
    ).astype(str)
    df_fe["Rush_Hour"] = df_fe["Hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df_fe["Late_Night"] = df_fe["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df_fe["Weekend"] = df_fe["Day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    df_fe["Day_Num"] = df_fe["Day_of_week"].map({"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}).fillna(-1).astype(int)

# Risk scores
age_score = {"Under 18": 5, "18-30": 3, "31-50": 2, "Over 51": 3, "Unknown": 2}
experience_score = {"No Licence": 5, "Below 1yr": 4, "1-2yr": 3, "2-5yr": 2, "5-10yr": 1, "Above 10yr": 1, "Unknown": 3}
vehicle_age_score = {"Below 1yr": 1, "1-2yr": 1, "2-5yrs": 2, "5-10yrs": 3, "Above 10yr": 4, "Unknown": 2}
weather_score = {"Normal": 1, "Raining": 3, "Fog or mist": 4, "Cloudy": 2, "Snow": 4, "Unknown": 2, "Windy": 2, "Raining and Windy": 3}
junction_score = {"No junction": 1, "Y Shape": 3, "Crossing": 3, "T Shape": 2, "Roundabout": 3, "Unknown": 2, "L Shape": 2, "O Shape": 2}
surface_score = {"Dry": 1, "Wet": 3, "Flood over 3cm. deep": 5, "Snow": 4, "Ice": 5, "Unknown": 2}
light_score = {"Daylight": 1, "Darkness - lights lit": 2, "Darkness - lights unlit": 4, "Darkness - no lighting": 5}

df_fe["Driver_Age_Risk"] = df_fe["Age_band_of_driver"].map(age_score).fillna(2)
df_fe["Driver_Experience_Score"] = df_fe["Driving_experience"].map(experience_score).fillna(3)
df_fe["Vehicle_Age_Risk"] = df_fe["Service_year_of_vehicle"].map(vehicle_age_score).fillna(2)
df_fe["Weather_Risk_Score"] = df_fe["Weather_conditions"].map(weather_score).fillna(2)
df_fe["Junction_Complexity"] = df_fe["Types_of_Junction"].map(junction_score).fillna(2)
df_fe["Road_Surface_Risk"] = df_fe["Road_surface_conditions"].map(surface_score).fillna(1)
df_fe["Light_Risk"] = df_fe["Light_conditions"].map(light_score).fillna(1)

df_fe["Driver_Risk_Index"] = 0.55 * df_fe["Driver_Experience_Score"] + 0.45 * df_fe["Driver_Age_Risk"]
df_fe["Environmental_Risk_Index"] = (df_fe["Weather_Risk_Score"] + df_fe["Road_Surface_Risk"] + df_fe["Light_Risk"] + df_fe["Junction_Complexity"]) / 4
df_fe["Accident_Risk_Index"] = 0.40 * df_fe["Driver_Risk_Index"] + 0.40 * df_fe["Environmental_Risk_Index"] + 0.20 * df_fe["Vehicle_Age_Risk"]

# Interaction features
df_fe["Casualty_per_vehicle"] = df_fe["Number_of_casualties"] / df_fe["Number_of_vehicles_involved"].clip(lower=1)
df_fe["Multiple_Casualties"] = (df_fe["Number_of_casualties"] >= 2).astype(int)
df_fe["Heavy_Traffic"] = (df_fe["Number_of_vehicles_involved"] >= 3).astype(int)
df_fe["Poor_Visibility"] = ((df_fe["Light_conditions"] != "Daylight") & (df_fe["Weather_conditions"] != "Normal")).astype(int)
df_fe["Wet_Night"] = ((df_fe["Road_surface_conditions"] != "Dry") & (df_fe["Light_conditions"] != "Daylight")).astype(int)
df_fe["High_Risk_Driver"] = ((df_fe["Age_band_of_driver"] == "Under 18") | (df_fe["Driving_experience"].isin(["No Licence", "Below 1yr"]))).astype(int)
df_fe["Experienced_Driver"] = df_fe["Driving_experience"].isin(["5-10yr", "Above 10yr"]).astype(int)
df_fe["Complex_Road"] = ((df_fe["Types_of_Junction"] != "No junction") & (df_fe["Lanes_or_Medians"] != "Undivided Two way")).astype(int)
df_fe["Old_Vehicle_Inexperienced_Driver"] = (df_fe["Service_year_of_vehicle"].isin(["Above 10yr", "5-10yrs"]) & df_fe["Driving_experience"].isin(["Below 1yr", "No Licence"])).astype(int)

# Context combinations
df_fe["Driver_Context"] = df_fe["Age_band_of_driver"] + "_" + df_fe["Driving_experience"]
df_fe["Road_Environment"] = df_fe["Road_surface_conditions"] + "_" + df_fe["Light_conditions"] + "_" + df_fe["Weather_conditions"]
df_fe["Junction_Lane_Context"] = df_fe["Types_of_Junction"] + "_" + df_fe["Lanes_or_Medians"]
df_fe["Vehicle_Driver_Context"] = df_fe["Type_of_vehicle"] + "_" + df_fe["Driving_experience"]
df_fe["Cause_Collision_Context"] = df_fe["Cause_of_accident"] + "_" + df_fe["Type_of_collision"]

# Frequency encoding
for col in df_fe.select_dtypes(include=["object", "string"]).columns:
    if col not in ["Accident_severity"]:
        freq_map = df_fe[col].value_counts(normalize=True)
        df_fe[f"{col}_Frequency"] = df_fe[col].map(freq_map).astype(float)

# Advanced interactions
df_fe["Night_Poor_Visibility"] = df_fe["Late_Night"] * df_fe["Poor_Visibility"]
df_fe["RushHour_HeavyTraffic"] = df_fe["Rush_Hour"] * df_fe["Heavy_Traffic"]
df_fe["WetRoad_Night"] = df_fe["Wet_Night"] * df_fe["Late_Night"]
df_fe["RiskDriver_Night"] = df_fe["High_Risk_Driver"] * df_fe["Late_Night"]
df_fe["ComplexRoad_PoorVisibility"] = df_fe["Complex_Road"] * df_fe["Poor_Visibility"]
df_fe["Driver_Road_Risk"] = df_fe["Driver_Risk_Index"] * df_fe["Road_Surface_Risk"]
df_fe["Driver_Junction_Risk"] = df_fe["Driver_Risk_Index"] * df_fe["Junction_Complexity"]
df_fe["Environment_Junction_Risk"] = df_fe["Environmental_Risk_Index"] * df_fe["Junction_Complexity"]
df_fe["Vehicle_Experience_Risk"] = df_fe["Vehicle_Age_Risk"] * (1 - df_fe["Driver_Experience_Score"])
df_fe["Composite_Risk_Score"] = (
    0.30 * df_fe["Driver_Risk_Index"] +
    0.25 * df_fe["Environmental_Risk_Index"] +
    0.20 * df_fe["Vehicle_Age_Risk"] +
    0.15 * df_fe["Road_Surface_Risk"] +
    0.10 * df_fe["Junction_Complexity"]
)
df_fe["Severe_Driving_Context"] = df_fe["High_Risk_Driver"] * df_fe["Poor_Visibility"] * df_fe["Rush_Hour"]
df_fe["Night_Complex_Road"] = df_fe["Late_Night"] * df_fe["Complex_Road"]

print(f"Shape after feature engineering: {df_fe.shape}")

# ============================================================
# 3. PREPARE X, y
# ============================================================
X_final = df_fe.drop(columns=["Accident_severity", "Time"])
y_final = df_fe["Accident_severity"]

# One-hot encode
X_final = pd.get_dummies(X_final, drop_first=True)

# Clean column names
X_final.columns = (
    X_final.columns.astype(str)
    .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_")
)
X_final = X_final.loc[:, ~X_final.columns.duplicated()]

print(f"Features after encoding: {X_final.shape[1]}")

# Handle NaN values
print("\nHandling NaN values...")
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_final), columns=X_final.columns)
print(f"NaN count after imputation: {X_imputed.isna().sum().sum()}")

# ============================================================
# 4. FEATURE SELECTION — keep top 80 features
# ============================================================
print("\nSelecting top 80 features...")
le = LabelEncoder()
y_encoded = le.fit_transform(y_final)

selector = SelectKBest(mutual_info_classif, k=80)
X_selected = selector.fit_transform(X_imputed, y_encoded)
selected_features = X_final.columns[selector.get_support()].tolist()

print(f"Selected {len(selected_features)} features")
print(f"Top features: {selected_features[:20]}")

# Save feature order for deployment
feature_order_new = selected_features

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed[selected_features], y_final,
    test_size=0.20, random_state=42, stratify=y_final
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Class distribution:\n{y_train.value_counts()}")

# ============================================================
# 6. COMPUTE CLASS WEIGHTS (inverse frequency)
# ============================================================
class_counts = y_train.value_counts()
total = len(y_train)
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
print(f"\nClass weights: {class_weights}")

# Apply weights to each sample
sample_weights = y_train.map(class_weights).values

# ============================================================
# 7. TRAIN XGBOOST WITH CLASS WEIGHTS
# ============================================================
print("\nTraining XGBoost with class weights...")

y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

xgb_new = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=5,
    gamma=0.2,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)

xgb_new.fit(X_train, y_train_enc, sample_weight=sample_weights)

# ============================================================
# 8. EVALUATE
# ============================================================
print("\n" + "="*60)
print("EVALUATION ON TEST SET")
print("="*60)

y_pred_enc = xgb_new.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)

print(classification_report(y_test, y_pred, zero_division=0))

# Fatal recall
fatal_idx = np.where(le.classes_ == "Fatal injury")[0][0]
fatal_recall = recall_score(y_test_enc, y_pred_enc, labels=[fatal_idx], average=None)[0]
print(f"FATAL RECALL: {fatal_recall*100:.1f}%")

# ============================================================
# 9. TEST WITH EXTREME FATAL CONDITIONS
# ============================================================
print("\n" + "="*60)
print("TESTING EXTREME FATAL CONDITIONS")
print("="*60)

# Create a test case that should be fatal
test_case = pd.DataFrame([{
    "Age_band_of_driver": "Under 18",
    "Sex_of_driver": "Male",
    "Driving_experience": "No Licence",
    "Type_of_vehicle": "Motorcycle",
    "Weather_conditions": "Fog or mist",
    "Road_surface_conditions": "Flood over 3cm. deep",
    "Light_conditions": "Darkness - no lighting",
    "Day_of_week": "Saturday",
    "Time": "23:30:00",
    "Number_of_casualties": 3,
    "Number_of_vehicles_involved": 2,
    "Types_of_Junction": "Y Shape",
    "Lanes_or_Medians": "Undivided Two way",
    "Cause_of_accident": "Driving carelessly",
    "Type_of_collision": "Vehicle with vehicle",
    "Area_accident_occured": " Rural",
    "Vehicle_movement": "Going straight",
    "Casualty_class": "Driver",
    "Sex_of_casualty": "Male",
    "Age_band_of_casualty": "Under 18",
    "Casualty_severity": "Fatal injury",
    "Work_of_casuality": "Unemployed",
    "Fitness_of_casuality": "Normal",
    "Pedestrian_movement": "Not a Pedestrian",
    "Owner_of_vehicle": "Private",
    "Service_year_of_vehicle": "Above 10yr",
    "Educational_level": "Junior high school",
    "Vehicle_driver_relation": "Driver",
    "Defect_of_vehicle": "No defect",
}])

# Apply same feature engineering
test_fe = test_case.copy()
test_fe["Time_Period"] = "Night"
test_fe["Driver_Age_Risk"] = test_fe["Age_band_of_driver"].map(age_score).fillna(2)
test_fe["Driver_Experience_Score"] = test_fe["Driving_experience"].map(experience_score).fillna(3)
test_fe["Vehicle_Age_Risk"] = test_fe["Service_year_of_vehicle"].map(vehicle_age_score).fillna(2)
test_fe["Weather_Risk_Score"] = test_fe["Weather_conditions"].map(weather_score).fillna(2)
test_fe["Junction_Complexity"] = test_fe["Types_of_Junction"].map(junction_score).fillna(2)
test_fe["Road_Surface_Risk"] = test_fe["Road_surface_conditions"].map(surface_score).fillna(1)
test_fe["Light_Risk"] = test_fe["Light_conditions"].map(light_score).fillna(1)
test_fe["Driver_Risk_Index"] = 0.55 * test_fe["Driver_Experience_Score"] + 0.45 * test_fe["Driver_Age_Risk"]
test_fe["Environmental_Risk_Index"] = (test_fe["Weather_Risk_Score"] + test_fe["Road_Surface_Risk"] + test_fe["Light_Risk"] + test_fe["Junction_Complexity"]) / 4
test_fe["Accident_Risk_Index"] = 0.40 * test_fe["Driver_Risk_Index"] + 0.40 * test_fe["Environmental_Risk_Index"] + 0.20 * test_fe["Vehicle_Age_Risk"]
test_fe["Casualty_per_vehicle"] = test_fe["Number_of_casualties"] / test_fe["Number_of_vehicles_involved"].clip(lower=1)
test_fe["Multiple_Casualties"] = (test_fe["Number_of_casualties"] >= 2).astype(int)
test_fe["Heavy_Traffic"] = (test_fe["Number_of_vehicles_involved"] >= 3).astype(int)
test_fe["Poor_Visibility"] = ((test_fe["Light_conditions"] != "Daylight") & (test_fe["Weather_conditions"] != "Normal")).astype(int)
test_fe["Wet_Night"] = ((test_fe["Road_surface_conditions"] != "Dry") & (test_fe["Light_conditions"] != "Daylight")).astype(int)
test_fe["High_Risk_Driver"] = ((test_fe["Age_band_of_driver"] == "Under 18") | (test_fe["Driving_experience"].isin(["No Licence", "Below 1yr"]))).astype(int)
test_fe["Experienced_Driver"] = test_fe["Driving_experience"].isin(["5-10yr", "Above 10yr"]).astype(int)
test_fe["Complex_Road"] = ((test_fe["Types_of_Junction"] != "No junction") & (test_fe["Lanes_or_Medians"] != "Undivided Two way")).astype(int)
test_fe["Old_Vehicle_Inexperienced_Driver"] = (test_fe["Service_year_of_vehicle"].isin(["Above 10yr", "5-10yrs"]) & test_fe["Driving_experience"].isin(["Below 1yr", "No Licence"])).astype(int)
test_fe["Rush_Hour"] = 0
test_fe["Late_Night"] = 1
test_fe["Driver_Context"] = test_fe["Age_band_of_driver"] + "_" + test_fe["Driving_experience"]
test_fe["Road_Environment"] = test_fe["Road_surface_conditions"] + "_" + test_fe["Light_conditions"] + "_" + test_fe["Weather_conditions"]
test_fe["Junction_Lane_Context"] = test_fe["Types_of_Junction"] + "_" + test_fe["Lanes_or_Medians"]
test_fe["Vehicle_Driver_Context"] = test_fe["Type_of_vehicle"] + "_" + test_fe["Driving_experience"]
test_fe["Cause_Collision_Context"] = test_fe["Cause_of_accident"] + "_" + test_fe["Type_of_collision"]
test_fe["Night_Poor_Visibility"] = test_fe["Late_Night"] * test_fe["Poor_Visibility"]
test_fe["RushHour_HeavyTraffic"] = test_fe["Rush_Hour"] * test_fe["Heavy_Traffic"]
test_fe["WetRoad_Night"] = test_fe["Wet_Night"] * test_fe["Late_Night"]
test_fe["RiskDriver_Night"] = test_fe["High_Risk_Driver"] * test_fe["Late_Night"]
test_fe["ComplexRoad_PoorVisibility"] = test_fe["Complex_Road"] * test_fe["Poor_Visibility"]
test_fe["Driver_Road_Risk"] = test_fe["Driver_Risk_Index"] * test_fe["Road_Surface_Risk"]
test_fe["Driver_Junction_Risk"] = test_fe["Driver_Risk_Index"] * test_fe["Junction_Complexity"]
test_fe["Environment_Junction_Risk"] = test_fe["Environmental_Risk_Index"] * test_fe["Junction_Complexity"]
test_fe["Vehicle_Experience_Risk"] = test_fe["Vehicle_Age_Risk"] * (1 - test_fe["Driver_Experience_Score"])
test_fe["Composite_Risk_Score"] = 0.30 * test_fe["Driver_Risk_Index"] + 0.25 * test_fe["Environmental_Risk_Index"] + 0.20 * test_fe["Vehicle_Age_Risk"] + 0.15 * test_fe["Road_Surface_Risk"] + 0.10 * test_fe["Junction_Complexity"]
test_fe["Severe_Driving_Context"] = test_fe["High_Risk_Driver"] * test_fe["Poor_Visibility"] * test_fe["Rush_Hour"]
test_fe["Night_Complex_Road"] = test_fe["Late_Night"] * test_fe["Complex_Road"]
test_fe["Day_Num"] = 5
test_fe["Weekend"] = 1

# Frequency encoding
for col in test_fe.select_dtypes(include=["object"]).columns:
    if col not in ["Accident_severity"] and f"{col}_Frequency" in df_fe.columns:
        test_fe[f"{col}_Frequency"] = df_fe[col].value_counts(normalize=True).get(test_fe[col].values[0], 0)

# One-hot encode
test_dummies = pd.get_dummies(test_fe, drop_first=True)
test_dummies.columns = (
    test_dummies.columns.astype(str)
    .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
    .str.strip("_")
)

# Align with training features
for col in selected_features:
    if col not in test_dummies.columns:
        test_dummies[col] = 0
test_dummies = test_dummies[selected_features]

# Predict
proba = xgb_new.predict_proba(test_dummies)[0]
pred_enc = xgb_new.predict(test_dummies)[0]
pred_label = le.inverse_transform([pred_enc])[0]

print(f"\nPrediction: {pred_label}")
print(f"Confidence: {proba[pred_enc]*100:.1f}%")
print(f"\nClass Probabilities:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {proba[i]*100:.1f}%")

# ============================================================
# 10. SAVE MODEL
# ============================================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save with new feature order
joblib.dump(xgb_new, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(selected_features, "feature_order.pkl")

print(f"model.pkl saved ({xgb_new.n_features_in_} features)")
print(f"label_encoder.pkl saved")
print(f"feature_order.pkl saved ({len(selected_features)} features)")
print("\nDone!")
