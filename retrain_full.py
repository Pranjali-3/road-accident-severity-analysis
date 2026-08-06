"""
Retrain XGBoost with ALL 738 features matching notebook exactly + class weights for fatal recall.
"""
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, recall_score

warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("Road.csv")

# ============================================================
# FEATURE ENGINEERING — matches notebook exactly
# ============================================================
df_fe = df.copy()

# --- TEMPORAL FEATURES ---
df_fe["Time"] = pd.to_datetime(df_fe["Time"], format="%H:%M:%S", errors="coerce")
df_fe["Hour"] = df_fe["Time"].dt.hour.fillna(0).astype(int)
df_fe["Minute"] = df_fe["Time"].dt.minute.fillna(0).astype(int)
df_fe["Time_Minutes"] = (df_fe["Hour"] * 60) + df_fe["Minute"]
df_fe["Hour_Sin"] = np.sin(2 * np.pi * df_fe["Hour"] / 24)
df_fe["Hour_Cos"] = np.cos(2 * np.pi * df_fe["Hour"] / 24)
df_fe["Rush_Hour"] = df_fe["Hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df_fe["Late_Night"] = df_fe["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

def get_time_period(hour):
    if hour < 6: return "Night"
    if hour < 12: return "Morning"
    if hour < 18: return "Afternoon"
    return "Evening"

df_fe["Time_Period"] = df_fe["Hour"].apply(get_time_period)

day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
df_fe["Day_Num"] = df_fe["Day_of_week"].map(day_map).fillna(-1).astype(int)
df_fe["Weekend"] = df_fe["Day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

# --- RISK SCORES ---
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

# --- INTERACTION FEATURES ---
df_fe["Casualty_per_vehicle"] = df_fe["Number_of_casualties"] / df_fe["Number_of_vehicles_involved"].clip(lower=1)
df_fe["Multiple_Casualties"] = (df_fe["Number_of_casualties"] >= 2).astype(int)
df_fe["Heavy_Traffic"] = (df_fe["Number_of_vehicles_involved"] >= 3).astype(int)
df_fe["Poor_Visibility"] = ((df_fe["Light_conditions"] != "Daylight") & (df_fe["Weather_conditions"] != "Normal")).astype(int)
df_fe["Wet_Night"] = ((df_fe["Road_surface_conditions"] != "Dry") & (df_fe["Light_conditions"] != "Daylight")).astype(int)
df_fe["High_Risk_Driver"] = ((df_fe["Age_band_of_driver"] == "Under 18") | (df_fe["Driving_experience"].isin(["No Licence", "Below 1yr"]))).astype(int)
df_fe["Experienced_Driver"] = df_fe["Driving_experience"].isin(["5-10yr", "Above 10yr"]).astype(int)
df_fe["Complex_Road"] = ((df_fe["Types_of_Junction"] != "No junction") & (df_fe["Lanes_or_Medians"] != "Undivided Two way")).astype(int)
df_fe["Old_Vehicle_Inexperienced_Driver"] = (df_fe["Service_year_of_vehicle"].isin(["Above 10yr", "5-10yrs"]) & df_fe["Driving_experience"].isin(["Below 1yr", "No Licence"])).astype(int)

# --- CONTEXT COMBINATIONS ---
df_fe["Driver_Context"] = df_fe["Age_band_of_driver"] + "_" + df_fe["Driving_experience"]
df_fe["Road_Environment"] = df_fe["Road_surface_conditions"] + "_" + df_fe["Light_conditions"] + "_" + df_fe["Weather_conditions"]
df_fe["Junction_Lane_Context"] = df_fe["Types_of_Junction"] + "_" + df_fe["Lanes_or_Medians"]
df_fe["Vehicle_Driver_Context"] = df_fe["Type_of_vehicle"] + "_" + df_fe["Driving_experience"]
df_fe["Cause_Collision_Context"] = df_fe["Cause_of_accident"] + "_" + df_fe["Type_of_collision"]
df_fe["Area_Time_Context"] = df_fe["Area_accident_occured"] + "_" + df_fe["Time_Period"]

# --- FREQUENCY ENCODING ---
for col in df_fe.select_dtypes(include=["object", "string"]).columns:
    if col not in ["Accident_severity"]:
        freq_map = df_fe[col].value_counts(normalize=True)
        df_fe[f"{col}_Frequency"] = df_fe[col].map(freq_map).astype(float)

# --- ADVANCED INTERACTIONS ---
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

print(f"Shape after feature engineering: {df_fe.shape}")

# ============================================================
# PREPARE X, y
# ============================================================
X_all = df_fe.drop(columns=["Accident_severity", "Time"])
y_all = df_fe["Accident_severity"]

X_all = pd.get_dummies(X_all, drop_first=True)
X_all.columns = X_all.columns.astype(str).str.replace(r"[^A-Za-z0-9_]", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")
X_all = X_all.loc[:, ~X_all.columns.duplicated()]

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_all), columns=X_all.columns)

le = LabelEncoder()
y_encoded = le.fit_transform(y_all)

print(f"Features after encoding: {X_imputed.shape[1]}")
print(f"Classes: {le.classes_}")

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_all, test_size=0.20, random_state=42, stratify=y_all)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

# ============================================================
# CLASS WEIGHTS — 78x for fatal, extra 3x boost
# ============================================================
class_counts = y_train.value_counts()
total = len(y_train)
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
class_weights["Fatal injury"] *= 3
sample_weights = y_train.map(class_weights).values

print(f"Class weights: {class_weights}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# TRAIN XGBOOST WITH ALL FEATURES + CLASS WEIGHTS
# ============================================================
print("\nTraining XGBoost...")

xgb_new = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1
)

xgb_new.fit(X_train, y_train_enc, sample_weight=sample_weights)

# ============================================================
# EVALUATE
# ============================================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

y_proba = xgb_new.predict_proba(X_test)

for threshold in [0.5, 0.2, 0.1, 0.05, 0.02]:
    preds = []
    for p in y_proba:
        if p[0] > threshold:
            preds.append(0)
        elif p[1] > 0.35:
            preds.append(1)
        else:
            preds.append(2)
    preds = np.array(preds)
    fatal_recall = recall_score(y_test_enc == 0, preds == 0)
    serious_recall = recall_score(y_test_enc == 1, preds == 1)
    accuracy = (preds == y_test_enc).mean()
    print(f"Threshold={threshold:.2f}: Fatal={fatal_recall*100:.1f}%, Serious={serious_recall*100:.1f}%, Acc={accuracy*100:.1f}%")

print("\n" + classification_report(y_test, le.inverse_transform(np.where(y_proba == y_proba.max(axis=1, keepdims=True), 1, 0).argmax(axis=1)), zero_division=0))

# ============================================================
# SAVE
# ============================================================
print("="*60)
print("SAVING")
print("="*60)

joblib.dump(xgb_new, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(list(X_imputed.columns), "feature_order.pkl")
joblib.dump(imputer, "imputer.pkl")

print(f"model.pkl: {xgb_new.n_features_in_} features")
print(f"feature_order.pkl: {len(X_imputed.columns)} features")
print(f"label_encoder.pkl: {list(le.classes_)}")
print("Done!")
