"""
Debug: predict the extreme risk scenario from the app.
"""
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.impute import SimpleImputer

# Load models
model_xgb = joblib.load("model.pkl")
model_lgb = joblib.load("model_lgb.pkl")
feature_order = joblib.load("feature_order.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Import MLP class from app
import sys
sys.path.insert(0, ".")
from app import AccidentSeverityMLP
mlp_model = AccidentSeverityMLP(input_dim=len(feature_order))
mlp_state = torch.load("model_mlp.pt", map_location="cpu", weights_only=True)
mlp_model.load_state_dict(mlp_state)
mlp_model.eval()

# Build user input (extreme risk scenario)
user_input = {
    "Day_of_week": "Tuesday",
    "Time": "09:30:00",
    "Number_of_vehicles_involved": 1,
    "Number_of_casualties": 1,
    "Casualty_severity": "Slight Injury",
    "Sex_of_driver": "Male",
    "Age_band_of_driver": "31-50",
    "Driving_experience": "5-10yr",
    "Type_of_vehicle": "Car",
    "Service_year_of_vehicle": "2-5yrs",
    "Weather_conditions": "Normal",
    "Road_surface_conditions": "Dry",
    "Light_conditions": "Daylight",
    "Types_of_Junction": "No junction",
    "Lanes_or_Medians": "Undivided Two way",
    "Type_of_collision": "Rollover",
    "Cause_of_accident": "Driving too close",
    "Pedestrian_movement": "Unknown",
    "Area_accident_occured": "Office areas",
}

user_df = pd.DataFrame([user_input])

# ---- FEATURE ENGINEERING (copy from app.py) ----
user_df["Time"] = pd.to_datetime(user_df["Time"], format="%H:%M:%S", errors="coerce")
user_df["Hour"] = user_df["Time"].dt.hour.fillna(0).astype(int)
user_df["Minute"] = user_df["Time"].dt.minute.fillna(0).astype(int)
user_df["Time_Minutes"] = user_df["Hour"] * 60 + user_df["Minute"]
user_df["Hour_Sin"] = np.sin(2 * np.pi * user_df["Hour"] / 24)
user_df["Hour_Cos"] = np.cos(2 * np.pi * user_df["Hour"] / 24)
user_df["Rush_Hour"] = user_df["Hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
user_df["Late_Night"] = user_df["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

def get_time_period(hour):
    if hour < 6: return "Night"
    if hour < 12: return "Morning"
    if hour < 18: return "Afternoon"
    return "Evening"

user_df["Time_Period"] = user_df["Hour"].apply(get_time_period)
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
user_df["Day_Num"] = user_df["Day_of_week"].map(day_map).fillna(-1).astype(int)
user_df["Weekend"] = user_df["Day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

age_score = {"Under 18": 5, "18-30": 3, "31-50": 2, "Over 51": 3, "Unknown": 2}
experience_score = {"No Licence": 5, "Below 1yr": 4, "1-2yr": 3, "2-5yr": 2, "5-10yr": 1, "Above 10yr": 1, "Unknown": 3}
vehicle_age_score = {"Below 1yr": 1, "1-2yr": 1, "2-5yrs": 2, "5-10yrs": 3, "Above 10yr": 4, "Unknown": 2}
weather_score = {"Normal": 1, "Raining": 3, "Fog or mist": 4, "Cloudy": 2, "Snow": 4, "Unknown": 2, "Windy": 2, "Raining and Windy": 3}
junction_score = {"No junction": 1, "Y Shape": 3, "Crossing": 3, "T Shape": 2, "Roundabout": 3, "Unknown": 2, "L Shape": 2, "O Shape": 2}
surface_score = {"Dry": 1, "Wet": 3, "Flood over 3cm. deep": 5, "Snow": 4, "Ice": 5, "Unknown": 2}
light_score = {"Daylight": 1, "Darkness - lights lit": 2, "Darkness - lights unlit": 4, "Darkness - no lighting": 5}

user_df["Driver_Age_Risk"] = user_df["Age_band_of_driver"].map(age_score).fillna(2)
user_df["Driver_Experience_Score"] = user_df["Driving_experience"].map(experience_score).fillna(3)
user_df["Vehicle_Age_Risk"] = user_df["Service_year_of_vehicle"].map(vehicle_age_score).fillna(2)
user_df["Weather_Risk_Score"] = user_df["Weather_conditions"].map(weather_score).fillna(2)
user_df["Junction_Complexity"] = user_df["Types_of_Junction"].map(junction_score).fillna(2)
user_df["Road_Surface_Risk"] = user_df["Road_surface_conditions"].map(surface_score).fillna(1)
user_df["Light_Risk"] = user_df["Light_conditions"].map(light_score).fillna(1)
user_df["Driver_Risk_Index"] = 0.55 * user_df["Driver_Experience_Score"] + 0.45 * user_df["Driver_Age_Risk"]
user_df["Environmental_Risk_Index"] = (user_df["Weather_Risk_Score"] + user_df["Road_Surface_Risk"] + user_df["Light_Risk"] + user_df["Junction_Complexity"]) / 4
user_df["Accident_Risk_Index"] = 0.40 * user_df["Driver_Risk_Index"] + 0.40 * user_df["Environmental_Risk_Index"] + 0.20 * user_df["Vehicle_Age_Risk"]
user_df["Casualty_per_vehicle"] = user_df["Number_of_casualties"] / user_df["Number_of_vehicles_involved"].clip(lower=1)
user_df["Multiple_Casualties"] = (user_df["Number_of_casualties"] >= 2).astype(int)
user_df["Heavy_Traffic"] = (user_df["Number_of_vehicles_involved"] >= 3).astype(int)
user_df["Poor_Visibility"] = ((user_df["Light_conditions"] != "Daylight") & (user_df["Weather_conditions"] != "Normal")).astype(int)
user_df["Wet_Night"] = ((user_df["Road_surface_conditions"] != "Dry") & (user_df["Light_conditions"] != "Daylight")).astype(int)
user_df["High_Risk_Driver"] = ((user_df["Age_band_of_driver"] == "Under 18") | (user_df["Driving_experience"].isin(["No Licence", "Below 1yr"]))).astype(int)
user_df["Experienced_Driver"] = user_df["Driving_experience"].isin(["5-10yr", "Above 10yr"]).astype(int)
user_df["Complex_Road"] = ((user_df["Types_of_Junction"] != "No junction") & (user_df["Lanes_or_Medians"] != "Undivided Two way")).astype(int)
user_df["Old_Vehicle_Inexperienced_Driver"] = (user_df["Service_year_of_vehicle"].isin(["Above 10yr", "5-10yrs"]) & user_df["Driving_experience"].isin(["Below 1yr", "No Licence"])).astype(int)
user_df["Driver_Context"] = user_df["Age_band_of_driver"] + "_" + user_df["Driving_experience"]
user_df["Road_Environment"] = user_df["Road_surface_conditions"] + "_" + user_df["Light_conditions"] + "_" + user_df["Weather_conditions"]
user_df["Junction_Lane_Context"] = user_df["Types_of_Junction"] + "_" + user_df["Lanes_or_Medians"]
user_df["Vehicle_Driver_Context"] = user_df["Type_of_vehicle"] + "_" + user_df["Driving_experience"]
user_df["Cause_Collision_Context"] = user_df["Cause_of_accident"] + "_" + user_df["Type_of_collision"]
user_df["Area_Time_Context"] = user_df["Area_accident_occured"] + "_" + user_df["Time_Period"]

for col in user_df.select_dtypes(include=["object", "string"]).columns:
    if col not in ["Accident_severity"]:
        freq_map = user_df[col].value_counts(normalize=True)
        user_df[f"{col}_Frequency"] = user_df[col].map(freq_map).astype(float)

user_df["Night_Poor_Visibility"] = user_df["Late_Night"] * user_df["Poor_Visibility"]
user_df["RushHour_HeavyTraffic"] = user_df["Rush_Hour"] * user_df["Heavy_Traffic"]
user_df["WetRoad_Night"] = user_df["Wet_Night"] * user_df["Late_Night"]
user_df["RiskDriver_Night"] = user_df["High_Risk_Driver"] * user_df["Late_Night"]
user_df["ComplexRoad_PoorVisibility"] = user_df["Complex_Road"] * user_df["Poor_Visibility"]
user_df["Driver_Road_Risk"] = user_df["Driver_Risk_Index"] * user_df["Road_Surface_Risk"]
user_df["Driver_Junction_Risk"] = user_df["Driver_Risk_Index"] * user_df["Junction_Complexity"]
user_df["Environment_Junction_Risk"] = user_df["Environmental_Risk_Index"] * user_df["Junction_Complexity"]
user_df["Vehicle_Experience_Risk"] = user_df["Vehicle_Age_Risk"] * (1 - user_df["Driver_Experience_Score"])
user_df["Composite_Risk_Score"] = 0.30 * user_df["Driver_Risk_Index"] + 0.25 * user_df["Environmental_Risk_Index"] + 0.20 * user_df["Vehicle_Age_Risk"] + 0.15 * user_df["Road_Surface_Risk"] + 0.10 * user_df["Junction_Complexity"]
user_df["Severe_Driving_Context"] = user_df["High_Risk_Driver"] * user_df["Poor_Visibility"] * user_df["Rush_Hour"]
user_df["Night_Complex_Road"] = user_df["Late_Night"] * user_df["Complex_Road"]

# One-hot encode
user_df = pd.get_dummies(user_df, drop_first=True)
user_df.columns = user_df.columns.astype(str).str.replace(r"[^A-Za-z0-9_]", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")
user_df = user_df.loc[:, ~user_df.columns.duplicated()]

# Align features
user_df = user_df.reindex(columns=feature_order, fill_value=0)

# Impute
user_df = pd.DataFrame(imputer.transform(user_df), columns=feature_order)

print(f"Features: {user_df.shape[1]} (expected {len(feature_order)})")

# ---- INDIVIDUAL MODEL PREDICTIONS ----
probs_xgb = model_xgb.predict_proba(user_df)[0]
probs_lgb = model_lgb.predict_proba(user_df)[0]

user_scaled = scaler.transform(user_df)
with torch.no_grad():
    logits = mlp_model(torch.tensor(user_scaled, dtype=torch.float32))
    probs_mlp = torch.softmax(logits, dim=1).numpy()[0]

classes = le.classes_
print(f"\n{'Model':<12} {'Fatal':>8} {'Serious':>8} {'Slight':>8}  Pred")
print("-" * 55)
for name, probs in [("XGBoost", probs_xgb), ("LightGBM", probs_lgb), ("MLP", probs_mlp)]:
    pred = classes[probs.argmax()]
    print(f"{name:<12} {probs[0]:>7.1%} {probs[1]:>7.1%} {probs[2]:>7.1%}  {pred}")

# Ensemble
w_xgb, w_lgb, w_mlp = 0.35, 0.45, 0.20
probs_blend = w_xgb * probs_xgb + w_lgb * probs_lgb + w_mlp * probs_mlp
print(f"\n{'Ensemble':<12} {probs_blend[0]:>7.1%} {probs_blend[1]:>7.1%} {probs_blend[2]:>7.1%}")

# Check thresholds
t_fatal, t_serious = 0.05, 0.35
print(f"\nThresholds: t_fatal={t_fatal}, t_serious={t_serious}")
print(f"Fatal prob {probs_blend[0]:.3f} vs t_fatal {t_fatal} -> {'FATAL' if probs_blend[0] > t_fatal else 'not fatal'}")
print(f"Serious prob {probs_blend[1]:.3f} vs t_serious {t_serious} -> {'SERIOUS' if probs_blend[1] > t_serious else 'not serious'}")

if probs_blend[0] > t_fatal:
    pred = 0
elif probs_blend[1] > t_serious:
    pred = 1
else:
    pred = 2

# Risk-based override
extreme_risk_count = sum([
    user_df["High_Risk_Driver"].values[0] == 1,
    user_df["Late_Night"].values[0] == 1,
    user_df["Poor_Visibility"].values[0] == 1,
    user_df["Wet_Night"].values[0] == 1,
    user_df["Heavy_Traffic"].values[0] == 1,
    user_df["Old_Vehicle_Inexperienced_Driver"].values[0] == 1,
    user_df["Complex_Road"].values[0] == 1,
    user_df["Severe_Driving_Context"].values[0] == 1,
])
print(f"\nExtreme risk factors: {extreme_risk_count}")
if extreme_risk_count >= 5:
    pred = 0
elif extreme_risk_count >= 4 and pred == 2:
    pred = 1

classes = ["Fatal injury", "Serious Injury", "Slight Injury"]
final = classes[pred]
print(f"Final prediction: {final}")
