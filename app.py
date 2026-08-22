import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="KSI Accident Severity Dashboard", page_icon="🚗", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(r"D:\jss_internship\ksi data.csv")

df = load_data()

model_xgb = joblib.load(r"D:\jss_internship\app_model_xgb.pkl")
model_lgbm = joblib.load(r"D:\jss_internship\app_model_lgbm.pkl")
scaler = joblib.load(r"D:\jss_internship\app_scaler.pkl")
label_encoder = joblib.load(r"D:\jss_internship\app_label_encoder.pkl")
feature_order = joblib.load(r"D:\jss_internship\app_feature_order.pkl")
cat_options = joblib.load(r"D:\jss_internship\app_cat_options.pkl")
cat_cols = joblib.load(r"D:\jss_internship\app_cat_cols.pkl")

w_xgb, w_lgb = 0.5, 0.5

page = st.sidebar.radio("Go To", ["Home", "Data Visualization", "Model Comparison", "Accident Prediction", "About"])

if page == "Home":
    st.title("🚗 Toronto KSI Accident Severity Prediction")
    st.markdown("---")
    st.header("Project Description")
    st.write("""
        This system predicts whether a road accident results in **Fatal** or **Non-Fatal** injury
        using Machine Learning on the Toronto KSI (Killed or Seriously Injured) dataset.

        Models tested:
        - Logistic Regression, Decision Tree, Random Forest
        - XGBoost, LightGBM, CatBoost
        - PyTorch MLP

        The deployed model is a **Blending Ensemble of XGBoost + LightGBM** achieving ~96% accuracy.
    """)
    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Features", df.shape[1])
    c3.metric("Fatal Injuries", int((df['acclass'] == 'Fatal Injury').sum()))
    c4.metric("Non-Fatal Injuries", int((df['acclass'] == 'Non-Fatal Injury').sum()))
    st.markdown("---")
    st.write("**Department:** Computer Science & Engineering (AI & ML)")
    st.write("**College:** JSS Academy of Technical Education, Noida")

elif page == "Data Visualization":
    st.title("📊 Data Visualization Dashboard")
    st.markdown("---")
    st.dataframe(df.head())
    st.markdown("---")
    chart = st.selectbox("Select Visualization", [
        "Accident Class Distribution", "Light Conditions", "Road Surface Conditions",
        "Road Class", "Vehicle Type", "Impact Type", "Driver Condition",
        "Road User Type", "Injury Distribution"
    ])
    if chart == "Accident Class Distribution":
        fig, ax = plt.subplots(figsize=(7, 5))
        df['acclass'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_title("Accident Class Distribution"); ax.set_ylabel("Count")
        st.pyplot(fig)
    elif chart == "Light Conditions":
        fig, ax = plt.subplots(figsize=(10, 5))
        df['light'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title("Accidents by Light Conditions"); plt.xticks(rotation=40, fontsize=8)
        st.pyplot(fig)
    elif chart == "Road Surface Conditions":
        fig, ax = plt.subplots(figsize=(10, 5))
        df['rdsfcond'].value_counts().plot(kind='bar', ax=ax, color='coral')
        ax.set_title("Accidents by Road Surface"); plt.xticks(rotation=30)
        st.pyplot(fig)
    elif chart == "Road Class":
        fig, ax = plt.subplots(figsize=(10, 5))
        df['road_class'].value_counts().plot(kind='bar', ax=ax, color='green')
        ax.set_title("Accidents by Road Class"); plt.xticks(rotation=45, fontsize=8)
        st.pyplot(fig)
    elif chart == "Vehicle Type":
        fig, ax = plt.subplots(figsize=(10, 5))
        df['vehtype'].value_counts().plot(kind='bar', ax=ax, color='purple')
        ax.set_title("Accidents by Vehicle Type"); plt.xticks(rotation=60, fontsize=7)
        st.pyplot(fig)
    elif chart == "Impact Type":
        fig, ax = plt.subplots(figsize=(10, 5))
        df['impactype'].value_counts().plot(kind='bar', ax=ax, color='orange')
        ax.set_title("Accidents by Impact Type"); plt.xticks(rotation=60, fontsize=7)
        st.pyplot(fig)
    elif chart == "Driver Condition":
        fig, ax = plt.subplots(figsize=(10, 5))
        df['drivcond'].value_counts().plot(kind='bar', ax=ax, color='red')
        ax.set_title("Accidents by Driver Condition"); plt.xticks(rotation=45, fontsize=8)
        st.pyplot(fig)
    elif chart == "Road User Type":
        fig, ax = plt.subplots(figsize=(8, 5))
        df['road_user'].value_counts().plot(kind='bar', ax=ax, color='teal')
        ax.set_title("Accidents by Road User"); plt.xticks(rotation=30)
        st.pyplot(fig)
    elif chart == "Injury Distribution":
        fig, ax = plt.subplots(figsize=(8, 5))
        df['injury'].value_counts().plot(kind='bar', ax=ax, color='crimson')
        ax.set_title("Injury Distribution"); plt.xticks(rotation=30)
        st.pyplot(fig)

elif page == "Model Comparison":
    st.title("🏆 Model Comparison")
    st.markdown("---")
    model_results = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost",
                   "RF Opt", "XGB Opt", "LGBM Opt", "CatOpt", "MLP", "XGB+ADASYN"],
        "Accuracy%": [78.41, 90.41, 95.74, 95.59, 95.62, 95.52, 95.67, 96.88, 96.83, 96.78, 96.10, 96.13],
        "Recall%": [62.12, 72.53, 73.04, 72.35, 72.70, 71.16, 73.04, 80.20, 80.03, 79.35, 83.45, 72.35],
    })
    st.dataframe(model_results.style.highlight_max(subset=["Accuracy%", "Recall%"], color="#90EE90"), use_container_width=True)
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(model_results))
    ax.bar(x - 0.2, model_results["Accuracy%"], 0.4, label="Accuracy", color="steelblue")
    ax.bar(x + 0.2, model_results["Recall%"], 0.4, label="Recall", color="coral")
    ax.set_xticks(x); ax.set_xticklabels(model_results["Model"], rotation=45, fontsize=9)
    ax.set_ylabel("Score (%)"); ax.set_title("All Models Comparison"); ax.legend()
    ax.set_ylim(0, 105); ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout(); st.pyplot(fig)

elif page == "Accident Prediction":
    st.title("🤖 Accident Severity Prediction")
    st.markdown("Enter accident details and click **Predict**.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Location & Road")
        stname1 = st.selectbox("Street Name 1", cat_options.get('stname1', []))
        stname2 = st.selectbox("Street Name 2", cat_options.get('stname2', []))
        traffictl = st.selectbox("Traffic Control", cat_options.get('traffictl', []))
        road_class = st.selectbox("Road Class", cat_options.get('road_class', []))
        rdsfcond = st.selectbox("Road Surface Condition", cat_options.get('rdsfcond', []))

    with col2:
        st.subheader("Light & Visibility")
        light = st.selectbox("Light Conditions", cat_options.get('light', []))
        visible = st.selectbox("Visibility", cat_options.get('visible', []))
        accloc = st.selectbox("Accident Location", cat_options.get('accloc', []))
        impactype = st.selectbox("Impact Type", cat_options.get('impactype', []))
        initdir = st.selectbox("Initial Direction", cat_options.get('initdir', []))

    with col3:
        st.subheader("Vehicle & Person")
        vehtype = st.selectbox("Vehicle Type", cat_options.get('vehtype', []))
        manoeuvre = st.selectbox("Manoeuvre", cat_options.get('manoeuvre', []))
        road_user = st.selectbox("Road User Type", cat_options.get('road_user', []))
        drivcond = st.selectbox("Driver Condition", cat_options.get('drivcond', []))
        drivact = st.selectbox("Driver Action", cat_options.get('drivact', []))

    col4, col5, col6 = st.columns(3)
    with col4:
        st.subheader("Injury & Safety")
        injury = st.selectbox("Injury Type", cat_options.get('injury', []))
        safequip = st.selectbox("Safety Equipment", cat_options.get('safequip', []))
    with col5:
        st.subheader("Ward & Division")
        wardname = st.selectbox("Ward", cat_options.get('wardname', []))
        division = st.selectbox("Division", cat_options.get('division', []))
    with col6:
        st.subheader("Neighbourhood")
        neighbourhood = st.selectbox("Neighbourhood", cat_options.get('neighbourhood', []))

    col7, col8, col9 = st.columns(3)
    with col7:
        st.subheader("Numbers")
        invage = st.slider("Age of Involved Person", 0, 100, 35)
        veh_no = st.number_input("Vehicle Number", 1, 10, 1)
        per_no = st.number_input("Person Number", 1, 10, 1)
        per_inv = st.number_input("Persons Involved", 1, 20, 1)
        longitude = st.number_input("Longitude", value=-79.38, format="%.6f")
        latitude = st.number_input("Latitude", value=43.71, format="%.6f")
    with col8:
        st.subheader("Date & Time")
        day_of_week = st.selectbox("Day of Week", list(range(7)),
                                   format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        month = st.selectbox("Month", list(range(1, 13)))
        year = st.selectbox("Year", list(range(2006, 2025)), index=14)
    with col9:
        st.subheader("Flags")
        aggressive = st.checkbox("Aggressive Driving")
        distracted = st.checkbox("Distracted Driving")
        red_light = st.checkbox("Red Light Violation")
        heavy_truck = st.checkbox("Heavy Truck Involved")
        failtorem = st.checkbox("Failed to Remain")
        cyclist = st.checkbox("Cyclist Involved")
        motorcyclist = st.checkbox("Motorcyclist Involved")
        other_micromobility = st.checkbox("Other Micromobility")
        older_adult = st.checkbox("Older Adult")
        pedestrian = st.checkbox("Pedestrian Involved")
        school_child = st.checkbox("School Child")

    st.markdown("---")
    predict = st.button("🚗 Predict Severity", use_container_width=True)

    if predict:
        user_df = pd.DataFrame([{
            'stname1': stname1, 'stname2': stname2, 'traffictl': traffictl,
            'road_class': road_class, 'rdsfcond': rdsfcond, 'light': light,
            'visible': visible, 'accloc': accloc, 'impactype': impactype,
            'initdir': initdir, 'vehtype': vehtype, 'manoeuvre': manoeuvre,
            'road_user': road_user, 'drivcond': drivcond, 'drivact': drivact,
            'injury': injury, 'safequip': safequip, 'wardname': wardname,
            'division': division, 'neighbourhood': neighbourhood,
            'invage': invage, 'veh_no': veh_no, 'per_no': per_no, 'per_inv': per_inv,
            'longitude': longitude, 'latitude': latitude,
            'day_of_week': day_of_week, 'month': month, 'year': year,
            'aggressive': int(aggressive), 'distracted': int(distracted),
            'red_light': int(red_light), 'heavy_truck': int(heavy_truck),
            'failtorem': int(failtorem), 'cyclist': int(cyclist),
            'motorcyclist': int(motorcyclist), 'other_micromobility': int(other_micromobility),
            'older_adult': int(older_adult), 'pedestrian': int(pedestrian),
            'school_child': int(school_child),
        }])

        for col in cat_cols:
            if col in user_df.columns:
                le = label_encoder[col]
                val = str(user_df[col].iloc[0])
                if val in le.classes_:
                    user_df[col] = le.transform([val])[0]
                else:
                    user_df[col] = 0

        user_df = user_df.fillna(0)
        user_df = user_df[feature_order]

        user_scaled = scaler.transform(user_df)

        probs_xgb = model_xgb.predict_proba(user_scaled)
        probs_lgb = model_lgbm.predict_proba(user_scaled)
        probs_blend = w_xgb * probs_xgb + w_lgb * probs_lgb
        proba = probs_blend[0]
        prediction = np.argmax(proba)
        predicted_label = ["Fatal Injury", "Non-Fatal Injury"][prediction]
        confidence = float(proba[prediction])

        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        if predicted_label == "Fatal Injury":
            st.error(f"### Predicted: **{predicted_label}**")
        else:
            st.success(f"### Predicted: **{predicted_label}**")
        st.metric("Confidence", f"{confidence:.1%}")

        st.subheader("Class Probabilities")
        proba_df = pd.DataFrame({
            "Severity": ["Fatal Injury", "Non-Fatal Injury"],
            "Probability": proba
        }).sort_values("Probability", ascending=False)
        fig_p, ax_p = plt.subplots(figsize=(6, 3))
        colors = ["#e74c3c" if p == proba.max() else "#3498db" for p in proba]
        ax_p.barh(proba_df["Severity"], proba_df["Probability"], color=colors)
        ax_p.set_xlabel("Probability"); ax_p.set_xlim(0, 1)
        for i, v in enumerate(proba_df["Probability"]):
            ax_p.text(v + 0.01, i, f"{v:.1%}", va="center")
        st.pyplot(fig_p)

        st.markdown("---")
        st.subheader("Risk Level")
        if predicted_label == "Fatal Injury":
            st.markdown("### 🔴 **HIGH RISK** — Fatal Injury Predicted")
        else:
            st.markdown("### 🟢 **LOWER RISK** — Non-Fatal Injury Predicted")

elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    st.write("""
        **Road Accident Severity Prediction System**

        Predicts Fatal vs Non-Fatal injury using ML on the Toronto KSI dataset.

        **Models:** XGBoost + LightGBM Blending Ensemble (~96% accuracy)

        **Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, LightGBM, Streamlit

        **Dataset:** Toronto KSI (Killed or Seriously Injured) — 50 features, 20,670 records
    """)
    st.markdown("---")
    st.success("Developed by Team: Pranjali • Mahi Gupta • Kanishka Patwal")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "KSI Accident Severity Prediction Dashboard • JSS Academy of Technical Education, Noida"
    "</div>", unsafe_allow_html=True
)
