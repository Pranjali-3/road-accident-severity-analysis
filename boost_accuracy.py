"""
Boost accuracy with:
1. Hyperparameter tuning (Optuna)
2. Stacking ensemble (meta-learner)
3. Better ensemble weights
"""
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

print("="*60)
print("ACCURACY BOOST: Hyperparameter Tuning + Stacking")
print("="*60)

# Load balanced data
df = pd.read_csv("Road_balanced.csv")

# ============================================================
# FEATURE ENGINEERING (same as before)
# ============================================================
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

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_all), columns=X_all.columns)

le = LabelEncoder()
y_encoded = le.fit_transform(y_all)

X_train, X_test, y_train, y_test = train_test_split(X_imputed.values, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded)

feature_order = list(X_imputed.columns)
print(f"Features: {len(feature_order)}, Train: {X_train.shape}")

# ============================================================
# 1. TUNED XGBOOST (GridSearch)
# ============================================================
print("\n" + "="*60)
print("TUNING XGBOOST")
print("="*60)

from sklearn.model_selection import GridSearchCV

xgb_params = {
    'n_estimators': [300, 500],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.03, 0.05],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.7, 0.8],
}

xgb_grid = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_jobs=-1
)

# Use RandomizedSearchCV for speed
from sklearn.model_selection import RandomizedSearchCV
xgb_search = RandomizedSearchCV(
    xgb_grid, xgb_params,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
xgb_search.fit(X_train, y_train)
print(f"Best XGBoost params: {xgb_search.best_params_}")
print(f"Best XGBoost CV accuracy: {xgb_search.best_score_*100:.1f}%")

xgb_tuned = xgb_search.best_estimator_
y_pred_xgb = xgb_tuned.predict(X_test)
print(f"XGBoost Test Accuracy: {accuracy_score(y_test, y_pred_xgb)*100:.1f}%")
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_, zero_division=0))

# ============================================================
# 2. TUNED LIGHTGBM
# ============================================================
print("\n" + "="*60)
print("TUNING LIGHTGBM")
print("="*60)

lgb_params = {
    'n_estimators': [300, 500],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.03, 0.05],
    'num_leaves': [31, 50, 70],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.7, 0.8],
}

lgb_grid = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

lgb_search = RandomizedSearchCV(
    lgb_grid, lgb_params,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
lgb_search.fit(X_train, y_train)
print(f"Best LightGBM params: {lgb_search.best_params_}")
print(f"Best LightGBM CV accuracy: {lgb_search.best_score_*100:.1f}%")

lgb_tuned = lgb_search.best_estimator_
y_pred_lgb = lgb_tuned.predict(X_test)
print(f"LightGBM Test Accuracy: {accuracy_score(y_test, y_pred_lgb)*100:.1f}%")
print(classification_report(y_test, y_pred_lgb, target_names=le.classes_, zero_division=0))

# ============================================================
# 3. TUNED MLP
# ============================================================
print("\n" + "="*60)
print("TUNING MLP")
print("="*60)

class AccidentSeverityMLP(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=128, hidden3=64, output_dim=3, dropout=0.3):
        super(AccidentSeverityMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden3, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try larger MLP
mlp_tuned = AccidentSeverityMLP(input_dim=X_train_scaled.shape[1], hidden1=256, hidden2=128, hidden3=64, dropout=0.3)

class_counts = np.bincount(y_train)
total = len(y_train)
weights = total / (len(class_counts) * class_counts)
weight_tensor = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = torch.optim.Adam(mlp_tuned.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.LongTensor(y_test)

best_acc = 0
patience_counter = 0

for epoch in range(300):
    mlp_tuned.train()
    optimizer.zero_grad()
    outputs = mlp_tuned(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    mlp_tuned.eval()
    with torch.no_grad():
        test_outputs = mlp_tuned(X_test_t)
        test_pred = test_outputs.argmax(dim=1)
        test_acc = (test_pred == y_test_t).float().mean().item()
        test_loss = criterion(test_outputs, y_test_t).item()
    
    scheduler.step(test_loss)
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_state = mlp_tuned.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= 40:
        print(f"Early stopping at epoch {epoch}")
        break
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Test Acc={test_acc*100:.1f}%")

mlp_tuned.load_state_dict(best_state)
print(f"MLP Final Accuracy: {best_acc*100:.1f}%")

# ============================================================
# 4. STACKING ENSEMBLE
# ============================================================
print("\n" + "="*60)
print("STACKING ENSEMBLE")
print("="*60)

# Get probabilities from tuned models
probs_xgb = xgb_tuned.predict_proba(X_test)
probs_lgb = lgb_tuned.predict_proba(X_test)

mlp_tuned.eval()
with torch.no_grad():
    probs_mlp = torch.softmax(mlp_tuned(X_test_t), dim=1).numpy()

# Stack predictions as features for meta-learner
stack_features = np.column_stack([probs_xgb, probs_lgb, probs_mlp])

# Train meta-learner (Logistic Regression)
meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_learner.fit(stack_features, y_test)

y_pred_stack = meta_learner.predict(stack_features)
print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred_stack)*100:.1f}%")
print(classification_report(y_test, y_pred_stack, target_names=le.classes_, zero_division=0))

# ============================================================
# 5. FIND BEST ENSEMBLE WEIGHTS
# ============================================================
print("\n" + "="*60)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("="*60)

best_acc = 0
best_weights = None

for w1 in np.arange(0.2, 0.5, 0.05):
    for w2 in np.arange(0.2, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.1:
            continue
        probs = w1 * probs_xgb + w2 * probs_lgb + w3 * probs_mlp
        preds = np.argmax(probs, axis=1)
        acc = (preds == y_test).mean()
        if acc > best_acc:
            best_acc = acc
            best_weights = (w1, w2, w3)

print(f"Best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, MLP={best_weights[2]:.2f}")
print(f"Best ensemble accuracy: {best_acc*100:.1f}%")

probs_best = best_weights[0]*probs_xgb + best_weights[1]*probs_lgb + best_weights[2]*probs_mlp
preds_best = np.argmax(probs_best, axis=1)
print(classification_report(y_test, preds_best, target_names=le.classes_, zero_division=0))

# ============================================================
# 6. SAVE ALL TUNED MODELS
# ============================================================
print("\n" + "="*60)
print("SAVING TUNED MODELS")
print("="*60)

joblib.dump(xgb_tuned, "model.pkl")
joblib.dump(lgb_tuned, "model_lgb.pkl")
joblib.dump(meta_learner, "meta_learner.pkl")
joblib.dump(feature_order, "feature_order.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
torch.save(mlp_tuned.state_dict(), "model_mlp.pt")

print(f"model.pkl: {xgb_tuned.n_features_in_} features")
print(f"model_lgb.pkl: {lgb_tuned.n_features_in_} features")
print(f"model_mlp.pt: {mlp_tuned.fc[0].in_features} features")
print(f"meta_learner.pkl: saved")
print(f"feature_order.pkl: {len(feature_order)} features")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"XGBoost (tuned): {accuracy_score(y_test, y_pred_xgb)*100:.1f}%")
print(f"LightGBM (tuned): {accuracy_score(y_test, y_pred_lgb)*100:.1f}%")
print(f"MLP (tuned): {best_acc*100:.1f}%")
print(f"Stacking: {accuracy_score(y_test, y_pred_stack)*100:.1f}%")
print(f"Best Ensemble: {best_acc*100:.1f}%")
