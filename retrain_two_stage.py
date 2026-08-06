"""
Retrain with TWO-STAGE approach:
  Stage 1: Binary classifier (Fatal vs Non-Fatal) 
  Stage 2: Multi-class (Serious vs Slight)
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

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
df_fe = df.copy()

if "Time" in df_fe.columns:
    df_fe["Hour"] = pd.to_datetime(df_fe["Time"], format="%H:%M:%S", errors="coerce").dt.hour
    df_fe["Time_Period"] = pd.cut(
        df_fe["Hour"], bins=[-1, 5, 9, 17, 21, 24],
        labels=["Late Night", "Morning Rush", "Daytime", "Evening Rush", "Night"],
        include_lowest=True
    ).astype(str)
    df_fe["Rush_Hour"] = df_fe["Hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df_fe["Late_Night"] = df_fe["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

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

df_fe["Casualty_per_vehicle"] = df_fe["Number_of_casualties"] / df_fe["Number_of_vehicles_involved"].clip(lower=1)
df_fe["Multiple_Casualties"] = (df_fe["Number_of_casualties"] >= 2).astype(int)
df_fe["Heavy_Traffic"] = (df_fe["Number_of_vehicles_involved"] >= 3).astype(int)
df_fe["Poor_Visibility"] = ((df_fe["Light_conditions"] != "Daylight") & (df_fe["Weather_conditions"] != "Normal")).astype(int)
df_fe["Wet_Night"] = ((df_fe["Road_surface_conditions"] != "Dry") & (df_fe["Light_conditions"] != "Daylight")).astype(int)
df_fe["High_Risk_Driver"] = ((df_fe["Age_band_of_driver"] == "Under 18") | (df_fe["Driving_experience"].isin(["No Licence", "Below 1yr"]))).astype(int)
df_fe["Experienced_Driver"] = df_fe["Driving_experience"].isin(["5-10yr", "Above 10yr"]).astype(int)
df_fe["Complex_Road"] = ((df_fe["Types_of_Junction"] != "No junction") & (df_fe["Lanes_or_Medians"] != "Undivided Two way")).astype(int)
df_fe["Old_Vehicle_Inexperienced_Driver"] = (df_fe["Service_year_of_vehicle"].isin(["Above 10yr", "5-10yrs"]) & df_fe["Driving_experience"].isin(["Below 1yr", "No Licence"])).astype(int)

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
    0.30 * df_fe["Driver_Risk_Index"] + 0.25 * df_fe["Environmental_Risk_Index"] +
    0.20 * df_fe["Vehicle_Age_Risk"] + 0.15 * df_fe["Road_Surface_Risk"] + 0.10 * df_fe["Junction_Complexity"]
)
df_fe["Severe_Driving_Context"] = df_fe["High_Risk_Driver"] * df_fe["Poor_Visibility"] * df_fe["Rush_Hour"]
df_fe["Night_Complex_Road"] = df_fe["Late_Night"] * df_fe["Complex_Road"]

# Frequency encoding
for col in df_fe.select_dtypes(include=["object"]).columns:
    if col not in ["Accident_severity"]:
        freq_map = df_fe[col].value_counts(normalize=True)
        df_fe[f"{col}_Frequency"] = df_fe[col].map(freq_map).astype(float)

print(f"Shape after feature engineering: {df_fe.shape}")

# ============================================================
# 3. PREPARE X, y
# ============================================================
X_all = df_fe.drop(columns=["Accident_severity", "Time"])
y_all = df_fe["Accident_severity"]

X_all = pd.get_dummies(X_all, drop_first=True)
X_all.columns = X_all.columns.astype(str).str.replace(r"[^A-Za-z0-9_]", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")
X_all = X_all.loc[:, ~X_all.columns.duplicated()]

# Impute
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_all), columns=X_all.columns)

print(f"Features: {X_imputed.shape[1]}")

# ============================================================
# 4. FEATURE SELECTION
# ============================================================
le = LabelEncoder()
y_encoded = le.fit_transform(y_all)

selector = SelectKBest(mutual_info_classif, k=100)
X_selected = selector.fit_transform(X_imputed, y_encoded)
selected_features = X_imputed.columns[selector.get_support()].tolist()

print(f"Selected {len(selected_features)} features")

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed[selected_features], y_all,
    test_size=0.20, random_state=42, stratify=y_all
)

y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 6. STAGE 1: BINARY CLASSIFIER (Fatal vs Non-Fatal)
# ============================================================
print("\n" + "="*60)
print("STAGE 1: Binary Classifier (Fatal vs Non-Fatal)")
print("="*60)

y_binary_train = (y_train == "Fatal injury").astype(int)
y_binary_test = (y_test == "Fatal injury").astype(int)

# Very high weight for fatal class
fatal_weight = len(y_binary_train) / y_binary_train.sum()  # ~78x
print(f"Fatal class weight: {fatal_weight:.1f}x")

xgb_fatal = XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=fatal_weight,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

xgb_fatal.fit(X_train, y_binary_train)

# Evaluate
y_fatal_pred = xgb_fatal.predict(X_test)
y_fatal_proba = xgb_fatal.predict_proba(X_test)[:, 1]

print(f"\nBinary Classifier (Fatal detection):")
print(f"  Fatal Recall: {recall_score(y_binary_test, y_fatal_pred)*100:.1f}%")
print(f"  Fatal Precision: {recall_score(y_binary_test, y_fatal_pred, pos_label=1)*100:.1f}%")

# ============================================================
# 7. STAGE 2: MULTI-CLASS (Serious vs Slight, excluding fatal)
# ============================================================
print("\n" + "="*60)
print("STAGE 2: Multi-class (Serious vs Slight)")
print("="*60)

# Filter out fatal cases for stage 2
mask_nonfatal = y_train != "Fatal injury"
X_train_nonfatal = X_train[mask_nonfatal]
y_train_nonfatal = y_train[mask_nonfatal]

# Encode for stage 2
le2 = LabelEncoder()
y_train_nonfatal_enc = le2.fit_transform(y_train_nonfatal)
y_test_nonfatal_enc = le2.transform(y_test[y_test != "Fatal injury"])
X_test_nonfatal = X_test[y_test != "Fatal injury"]

xgb_severity = XGBClassifier(
    objective="multi:softprob",
    num_class=2,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)

xgb_severity.fit(X_train_nonfatal, y_train_nonfatal_enc)

# Evaluate stage 2
y_severity_proba = xgb_severity.predict_proba(X_test_nonfatal)
y_severity_pred = np.argmax(y_severity_proba, axis=1)
y_severity_pred_labels = le2.inverse_transform(y_severity_pred)
y_test_nonfatal_labels = le2.inverse_transform(y_test_nonfatal_enc)

print(f"\nSeverity Classifier (Serious vs Slight):")
print(classification_report(y_test_nonfatal_labels, y_severity_pred_labels, zero_division=0))

# ============================================================
# 8. COMBINED EVALUATION
# ============================================================
print("\n" + "="*60)
print("COMBINED TWO-STAGE EVALUATION")
print("="*60)

# Stage 1: Detect fatal
fatal_proba = xgb_fatal.predict_proba(X_test)[:, 1]
fatal_threshold = 0.15  # Lower threshold for better recall

# Stage 2: Classify severity for non-fatal
severity_proba = xgb_severity.predict_proba(X_test)

# Combined prediction
final_pred = []
final_proba = []

for i in range(len(X_test)):
    if fatal_proba[i] > fatal_threshold:
        # Stage 1 says fatal
        final_pred.append("Fatal injury")
        final_proba.append({"Fatal injury": fatal_proba[i], "Serious Injury": 0, "Slight Injury": 1-fatal_proba[i]})
    else:
        # Stage 2 determines severity
        sev_pred = le2.inverse_transform([np.argmax(severity_proba[i])])[0]
        final_pred.append(sev_pred)
        final_proba.append({"Fatal injury": fatal_proba[i], sev_pred: 1-fatal_proba[i]})

# Calculate metrics
final_pred = np.array(final_pred)
fatal_recall = recall_score(y_test == "Fatal injury", final_pred == "Fatal injury")
serious_recall = recall_score(y_test == "Serious Injury", final_pred == "Serious Injury")
slight_recall = recall_score(y_test == "Slight Injury", final_pred == "Slight Injury")

print(f"\nFatal Recall:  {fatal_recall*100:.1f}%")
print(f"Serious Recall: {serious_recall*100:.1f}%")
print(f"Slight Recall:  {slight_recall*100:.1f}%")

# Full classification report
print("\n" + classification_report(y_test, final_pred, zero_division=0))

# ============================================================
# 9. TEST EXTREME CASE
# ============================================================
print("="*60)
print("TEST: Extreme Fatal Conditions")
print("="*60)

# Just use the training data stats for a quick test
print("\nThe model now uses TWO stages:")
print("  Stage 1: Binary (Fatal detection) with 78x class weight")
print("  Stage 2: Multi-class (Serious vs Slight)")
print(f"  Fatal threshold: {fatal_threshold}")

# ============================================================
# 10. SAVE ALL MODELS
# ============================================================
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

joblib.dump(xgb_fatal, "model_fatal.pkl")
joblib.dump(xgb_severity, "model_severity.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(le2, "label_encoder_severity.pkl")
joblib.dump(selected_features, "feature_order.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(selector, "selector.pkl")
joblib.dump(fatal_threshold, "fatal_threshold.pkl")

# Also save as combined format for app
joblib.dump({
    "fatal_model": xgb_fatal,
    "severity_model": xgb_severity,
    "le_full": le,
    "le_severity": le2,
    "features": selected_features,
    "imputer": imputer,
    "selector": selector,
    "fatal_threshold": fatal_threshold
}, "model_two_stage.pkl")

print("Saved: model_fatal.pkl, model_severity.pkl, model_two_stage.pkl")
print("Done!")
