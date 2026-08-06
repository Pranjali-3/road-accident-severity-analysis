"""Quick test: ensemble accuracy with new weights on the full test set."""
import pandas as pd
import numpy as np
import joblib, torch, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")

df = pd.read_csv("Road.csv")

# ---- FEATURE ENGINEERING (same pipeline) ----
df_fe = df.copy()
df_fe["Time"] = pd.to_datetime(df_fe["Time"], format="%H:%M:%S", errors="coerce")
df_fe["Hour"] = df_fe["Time"].dt.hour.fillna(0).astype(int)
df_fe["Minute"] = df_fe["Time"].dt.minute.fillna(0).astype(int)
df_fe["Time_Minutes"] = df_fe["Hour"]*60+df_fe["Minute"]
df_fe["Hour_Sin"] = np.sin(2*np.pi*df_fe["Hour"]/24)
df_fe["Hour_Cos"] = np.cos(2*np.pi*df_fe["Hour"]/24)
df_fe["Rush_Hour"] = df_fe["Hour"].isin([7,8,9,16,17,18,19]).astype(int)
df_fe["Late_Night"] = df_fe["Hour"].isin([22,23,0,1,2,3,4,5]).astype(int)
def get_tp(h):
    if h<6: return "Night"
    if h<12: return "Morning"
    if h<18: return "Afternoon"
    return "Evening"
df_fe["Time_Period"] = df_fe["Hour"].apply(get_tp)
day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
df_fe["Day_Num"] = df_fe["Day_of_week"].map(day_map).fillna(-1).astype(int)
df_fe["Weekend"] = df_fe["Day_of_week"].isin(["Saturday","Sunday"]).astype(int)
age_score = {"Under 18":5,"18-30":3,"31-50":2,"Over 51":3,"Unknown":2}
exp_score = {"No Licence":5,"Below 1yr":4,"1-2yr":3,"2-5yr":2,"5-10yr":1,"Above 10yr":1,"Unknown":3}
veh_score = {"Below 1yr":1,"1-2yr":1,"2-5yrs":2,"5-10yrs":3,"Above 10yr":4,"Unknown":2}
wth_score = {"Normal":1,"Raining":3,"Fog or mist":4,"Cloudy":2,"Snow":4,"Unknown":2,"Windy":2,"Raining and Windy":3}
jnc_score = {"No junction":1,"Y Shape":3,"Crossing":3,"T Shape":2,"Roundabout":3,"Unknown":2,"L Shape":2,"O Shape":2}
srf_score = {"Dry":1,"Wet":3,"Flood over 3cm. deep":5,"Snow":4,"Ice":5,"Unknown":2}
lit_score = {"Daylight":1,"Darkness - lights lit":2,"Darkness - lights unlit":4,"Darkness - no lighting":5}
df_fe["Driver_Age_Risk"] = df_fe["Age_band_of_driver"].map(age_score).fillna(2)
df_fe["Driver_Experience_Score"] = df_fe["Driving_experience"].map(exp_score).fillna(3)
df_fe["Vehicle_Age_Risk"] = df_fe["Service_year_of_vehicle"].map(veh_score).fillna(2)
df_fe["Weather_Risk_Score"] = df_fe["Weather_conditions"].map(wth_score).fillna(2)
df_fe["Junction_Complexity"] = df_fe["Types_of_Junction"].map(jnc_score).fillna(2)
df_fe["Road_Surface_Risk"] = df_fe["Road_surface_conditions"].map(srf_score).fillna(1)
df_fe["Light_Risk"] = df_fe["Light_conditions"].map(lit_score).fillna(1)
df_fe["Driver_Risk_Index"] = 0.55*df_fe["Driver_Experience_Score"]+0.45*df_fe["Driver_Age_Risk"]
df_fe["Environmental_Risk_Index"] = (df_fe["Weather_Risk_Score"]+df_fe["Road_Surface_Risk"]+df_fe["Light_Risk"]+df_fe["Junction_Complexity"])/4
df_fe["Accident_Risk_Index"] = 0.40*df_fe["Driver_Risk_Index"]+0.40*df_fe["Environmental_Risk_Index"]+0.20*df_fe["Vehicle_Age_Risk"]
df_fe["Casualty_per_vehicle"] = df_fe["Number_of_casualties"]/df_fe["Number_of_vehicles_involved"].clip(lower=1)
df_fe["Multiple_Casualties"] = (df_fe["Number_of_casualties"]>=2).astype(int)
df_fe["Heavy_Traffic"] = (df_fe["Number_of_vehicles_involved"]>=3).astype(int)
df_fe["Poor_Visibility"] = ((df_fe["Light_conditions"]!="Daylight")&(df_fe["Weather_conditions"]!="Normal")).astype(int)
df_fe["Wet_Night"] = ((df_fe["Road_surface_conditions"]!="Dry")&(df_fe["Light_conditions"]!="Daylight")).astype(int)
df_fe["High_Risk_Driver"] = ((df_fe["Age_band_of_driver"]=="Under 18")|(df_fe["Driving_experience"].isin(["No Licence","Below 1yr"]))).astype(int)
df_fe["Experienced_Driver"] = df_fe["Driving_experience"].isin(["5-10yr","Above 10yr"]).astype(int)
df_fe["Complex_Road"] = ((df_fe["Types_of_Junction"]!="No junction")&(df_fe["Lanes_or_Medians"]!="Undivided Two way")).astype(int)
df_fe["Old_Vehicle_Inexperienced_Driver"] = (df_fe["Service_year_of_vehicle"].isin(["Above 10yr","5-10yrs"])&df_fe["Driving_experience"].isin(["Below 1yr","No Licence"])).astype(int)
df_fe["Driver_Context"] = df_fe["Age_band_of_driver"]+"_"+df_fe["Driving_experience"]
df_fe["Road_Environment"] = df_fe["Road_surface_conditions"]+"_"+df_fe["Light_conditions"]+"_"+df_fe["Weather_conditions"]
df_fe["Junction_Lane_Context"] = df_fe["Types_of_Junction"]+"_"+df_fe["Lanes_or_Medians"]
df_fe["Vehicle_Driver_Context"] = df_fe["Type_of_vehicle"]+"_"+df_fe["Driving_experience"]
df_fe["Cause_Collision_Context"] = df_fe["Cause_of_accident"]+"_"+df_fe["Type_of_collision"]
df_fe["Area_Time_Context"] = df_fe["Area_accident_occured"]+"_"+df_fe["Time_Period"]
for col in df_fe.select_dtypes(include=["object","string"]).columns:
    if col not in ["Accident_severity"]:
        freq_map = df_fe[col].value_counts(normalize=True)
        df_fe[f"{col}_Frequency"] = df_fe[col].map(freq_map).astype(float)
df_fe["Night_Poor_Visibility"] = df_fe["Late_Night"]*df_fe["Poor_Visibility"]
df_fe["RushHour_HeavyTraffic"] = df_fe["Rush_Hour"]*df_fe["Heavy_Traffic"]
df_fe["WetRoad_Night"] = df_fe["Wet_Night"]*df_fe["Late_Night"]
df_fe["RiskDriver_Night"] = df_fe["High_Risk_Driver"]*df_fe["Late_Night"]
df_fe["ComplexRoad_PoorVisibility"] = df_fe["Complex_Road"]*df_fe["Poor_Visibility"]
df_fe["Driver_Road_Risk"] = df_fe["Driver_Risk_Index"]*df_fe["Road_Surface_Risk"]
df_fe["Driver_Junction_Risk"] = df_fe["Driver_Risk_Index"]*df_fe["Junction_Complexity"]
df_fe["Environment_Junction_Risk"] = df_fe["Environmental_Risk_Index"]*df_fe["Junction_Complexity"]
df_fe["Vehicle_Experience_Risk"] = df_fe["Vehicle_Age_Risk"]*(1-df_fe["Driver_Experience_Score"])
df_fe["Composite_Risk_Score"] = 0.30*df_fe["Driver_Risk_Index"]+0.25*df_fe["Environmental_Risk_Index"]+0.20*df_fe["Vehicle_Age_Risk"]+0.15*df_fe["Road_Surface_Risk"]+0.10*df_fe["Junction_Complexity"]
df_fe["Severe_Driving_Context"] = df_fe["High_Risk_Driver"]*df_fe["Poor_Visibility"]*df_fe["Rush_Hour"]
df_fe["Night_Complex_Road"] = df_fe["Late_Night"]*df_fe["Complex_Road"]

X_all = df_fe.drop(columns=["Accident_severity","Time"])
y_all = df_fe["Accident_severity"]
X_all = pd.get_dummies(X_all, drop_first=True)
X_all.columns = X_all.columns.astype(str).str.replace(r"[^A-Za-z0-9_]","_",regex=True).str.replace(r"_+","_",regex=True).str.strip("_")
X_all = X_all.loc[:,~X_all.columns.duplicated()]

imputer = joblib.load("imputer.pkl")
feature_order = joblib.load("feature_order.pkl")
for col in feature_order:
    if col not in X_all.columns:
        X_all[col] = 0
X_all = X_all[feature_order]
X_imputed = pd.DataFrame(imputer.transform(X_all), columns=feature_order)

le = joblib.load("label_encoder.pkl")
y_encoded = le.fit_transform(y_all)

X_train, X_test, y_train, y_test = train_test_split(X_imputed.values, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)

# Load models
model_xgb = joblib.load("model.pkl")
model_lgb = joblib.load("model_lgb.pkl")
scaler = joblib.load("scaler.pkl")

from app import AccidentSeverityMLP
mlp_model = AccidentSeverityMLP(input_dim=len(feature_order))
mlp_model.load_state_dict(torch.load("model_mlp.pt", map_location="cpu"))
mlp_model.eval()

# Predict
probs_xgb = model_xgb.predict_proba(X_test)
probs_lgb = model_lgb.predict_proba(X_test)
X_scaled = scaler.transform(X_test)
with torch.no_grad():
    probs_mlp = torch.softmax(mlp_model(torch.tensor(X_scaled, dtype=torch.float32)), dim=1).numpy()

# Test both weight sets
for name, w in [("Old 0.35/0.45/0.20", (0.35,0.45,0.20)), ("New 0.20/0.30/0.50", (0.20,0.30,0.50))]:
    probs = w[0]*probs_xgb + w[1]*probs_lgb + w[2]*probs_mlp
    preds = np.argmax(probs, axis=1)

    # With thresholds
    t_f, t_s = 0.05, 0.35
    final = np.full(len(probs), 2)
    final[probs[:,0] > t_f] = 0
    final[(probs[:,0] <= t_f) & (probs[:,1] > t_s)] = 1

    print(f"\n{'='*50}")
    print(f"WEIGHTS: {name}")
    print(f"{'='*50}")
    acc = (final == y_test).mean()
    print(f"Overall Accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, final, target_names=le.classes_, zero_division=0))
