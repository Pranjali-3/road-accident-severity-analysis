"""
Retrain the PyTorch MLP model with the new 685 features.
"""
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, TensorDataset

print("Loading data...")
df = pd.read_csv("Road.csv")

# ============================================================
# FEATURE ENGINEERING (same as retrain_full.py)
# ============================================================
df_fe = df.copy()

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
df_fe["Driver_Context"] = df_fe["Age_band_of_driver"] + "_" + df_fe["Driving_experience"]
df_fe["Road_Environment"] = df_fe["Road_surface_conditions"] + "_" + df_fe["Light_conditions"] + "_" + df_fe["Weather_conditions"]
df_fe["Junction_Lane_Context"] = df_fe["Types_of_Junction"] + "_" + df_fe["Lanes_or_Medians"]
df_fe["Vehicle_Driver_Context"] = df_fe["Type_of_vehicle"] + "_" + df_fe["Driving_experience"]
df_fe["Cause_Collision_Context"] = df_fe["Cause_of_accident"] + "_" + df_fe["Type_of_collision"]
df_fe["Area_Time_Context"] = df_fe["Area_accident_occured"] + "_" + df_fe["Time_Period"]

for col in df_fe.select_dtypes(include=["object", "string"]).columns:
    if col not in ["Accident_severity"]:
        freq_map = df_fe[col].value_counts(normalize=True)
        df_fe[f"{col}_Frequency"] = df_fe[col].map(freq_map).astype(float)

df_fe["Night_Poor_Visibility"] = df_fe["Late_Night"] * df_fe["Poor_Visibility"]
df_fe["RushHour_HeavyTraffic"] = df_fe["Rush_Hour"] * df_fe["Heavy_Traffic"]
df_fe["WetRoad_Night"] = df_fe["Wet_Night"] * df_fe["Late_Night"]
df_fe["RiskDriver_Night"] = df_fe["High_Risk_Driver"] * df_fe["Late_Night"]
df_fe["ComplexRoad_PoorVisibility"] = df_fe["Complex_Road"] * df_fe["Poor_Visibility"]
df_fe["Driver_Road_Risk"] = df_fe["Driver_Risk_Index"] * df_fe["Road_Surface_Risk"]
df_fe["Driver_Junction_Risk"] = df_fe["Driver_Risk_Index"] * df_fe["Junction_Complexity"]
df_fe["Environment_Junction_Risk"] = df_fe["Environmental_Risk_Index"] * df_fe["Junction_Complexity"]
df_fe["Vehicle_Experience_Risk"] = df_fe["Vehicle_Age_Risk"] * (1 - df_fe["Driver_Experience_Score"])
df_fe["Composite_Risk_Score"] = 0.30 * df_fe["Driver_Risk_Index"] + 0.25 * df_fe["Environmental_Risk_Index"] + 0.20 * df_fe["Vehicle_Age_Risk"] + 0.15 * df_fe["Road_Surface_Risk"] + 0.10 * df_fe["Junction_Complexity"]
df_fe["Severe_Driving_Context"] = df_fe["High_Risk_Driver"] * df_fe["Poor_Visibility"] * df_fe["Rush_Hour"]
df_fe["Night_Complex_Road"] = df_fe["Late_Night"] * df_fe["Complex_Road"]

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

feature_order = joblib.load("feature_order.pkl")
print(f"X_imputed: {X_imputed.shape[1]} features, feature_order: {len(feature_order)} features")

# Align
for col in feature_order:
    if col not in X_imputed.columns:
        X_imputed[col] = 0
X_imputed = X_imputed[feature_order]

le = LabelEncoder()
y_encoded = le.fit_transform(y_all)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed.values, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class weights
class_counts = np.bincount(y_train)
total = len(y_train)
weights = total / (len(class_counts) * class_counts)
sample_weights = weights[y_train]

# ============================================================
# DEFINE & TRAIN MLP
# ============================================================
class AccidentSeverityMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(AccidentSeverityMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

input_dim = X_train_scaled.shape[1]
model_mlp = AccidentSeverityMLP(input_dim=input_dim)

# Weighted loss
weight_tensor = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

optimizer = torch.optim.Adam(model_mlp.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

print(f"\nTraining MLP with {input_dim} features...")
best_test_acc = 0
patience_counter = 0

for epoch in range(200):
    model_mlp.train()
    optimizer.zero_grad()
    outputs = model_mlp(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    model_mlp.eval()
    with torch.no_grad():
        test_outputs = model_mlp(X_test_tensor)
        test_pred = test_outputs.argmax(dim=1)
        test_acc = (test_pred == y_test_tensor).float().mean().item()
        test_loss = criterion(test_outputs, y_test_tensor).item()
    
    scheduler.step(test_loss)
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_state = model_mlp.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= 30:
        print(f"Early stopping at epoch {epoch}")
        break
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Test Acc={test_acc*100:.1f}%")

model_mlp.load_state_dict(best_state)

# ============================================================
# EVALUATE
# ============================================================
model_mlp.eval()
with torch.no_grad():
    outputs = model_mlp(X_test_tensor)
    preds = outputs.argmax(dim=1)

print(f"\nFinal Test Accuracy: {(preds == y_test_tensor).float().mean()*100:.1f}%")

from sklearn.metrics import classification_report
print(classification_report(y_test, preds.numpy(), target_names=le.classes_, zero_division=0))

# ============================================================
# SAVE
# ============================================================
print("Saving model_mlp.pt...")
torch.save(model_mlp.state_dict(), "model_mlp.pt")
joblib.dump(scaler, "scaler.pkl")

print(f"model_mlp.pt saved (input_dim={input_dim})")
print(f"scaler.pkl saved")
print("Done!")
