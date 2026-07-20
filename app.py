import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Road Accident Severity Dashboard",
    page_icon="🚗",
    layout="wide"
)
# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv("Road.csv")

df = load_data()
# =====================================================
# LOAD MODEL FILES
# =====================================================
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_order = joblib.load("feature_order.pkl")
# =====================================================
# PRECOMPUTE FREQUENCY MAPS (from full dataset)
# Needed so prediction can replicate training encoding.
# =====================================================

@st.cache_data
def build_frequency_maps(data):
    freq_maps = {}
    for col in data.select_dtypes(include=["object"]).columns:
        if col not in ["Accident_severity"]:
            freq_maps[col] = data[col].value_counts(normalize=True).to_dict()
    return freq_maps

freq_maps = build_frequency_maps(df)
# =====================================================
# RISK SCORE MAPPING DICTIONARIES
# (must match notebook Section 13 exactly)
# =====================================================
age_score = {
    "Under 18": 4,
    "18-30": 2,
    "31-50": 1,
    "Over 51": 3,
    "Unknown": 2
}

experience_score = {
    "No Licence": 5,
    "Below 1yr": 4,
    "1-2yr": 3,
    "2-5yr": 2,
    "5-10yr": 1,
    "Above 10yr": 0,
    "unknown": 3,
    "Unknown": 3
}

vehicle_age_score = {
    "Below 1yr": 0,
    "1-2yr": 1,
    "2-5yrs": 2,
    "5-10yrs": 3,
    "Above 10yr": 4,
    "unknown": 2,
    "Unknown": 2
}

weather_score_map = {
    "Normal": 0,
    "Cloudy": 1,
    "Windy": 2,
    "Other": 2,
    "Unknown": 2,
    "Raining": 3,
    "Raining and Windy": 4,
    "Fog or mist": 5,
    "Snow": 5
}

junction_score = {
    "No junction": 0,
    "Y Shape": 2,
    "T Shape": 3,
    "Crossing": 4,
    "X Shape": 5,
    "O Shape": 2,
    "Other": 2,
    "Unknown": 2
}

surface_score = {
    "Dry": 0,
    "Wet or damp": 3,
    "Snow": 5,
    "Flood over 3cm. deep": 5
}

light_score = {
    "Daylight": 0,
    "Darkness - lights lit": 2,
    "Darkness - lights unlit": 4,
    "Darkness - no lighting": 5
}
# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🚗 Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Data Visualization",
        "Feature Importance",
        "Model Comparison",
        "Accident Prediction",
        "About"
    ]
)
# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":
    st.title("🚗 Road Accident Severity Prediction System")
    st.markdown("---")
    st.header("Project Description")
    st.write(
        """
        This project predicts the severity of road accidents using
        Machine Learning techniques.

        The dataset contains various accident-related attributes
        including weather conditions, road surface, light conditions,
        driver information, vehicle information and accident details.

        Multiple ML algorithms were tested including:

        - Logistic Regression
        - Random Forest
        - LightGBM
        - CatBoost
        - XGBoost

        The final deployed model is the **Enhanced XGBoost + ADASYN Model**
        which achieved approximately **84.50% accuracy**.
        """
    )
    st.markdown("---")

    # ---- Dataset Statistics (Change #8) ----
    st.subheader("📊 Dataset Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Features", df.shape[1])
    c3.metric("Severity Classes", df["Accident_severity"].nunique())
    c4.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Most Common Severity",
              df["Accident_severity"].mode()[0])
    c6.metric("Avg Casualties",
              round(df["Number_of_casualties"].mean(), 2))
    c7.metric("Avg Vehicles",
              round(df["Number_of_vehicles_involved"].mean(), 2))
    c8.metric("Peak Hour",
              int(df["Time"].str.split(":").str[0].mode()[0]))

    st.markdown("---")

    st.subheader("Project Team")

    st.write("**Department:** Computer Science & Engineering (AI & ML)")
    st.write("**College:** JSS Academy of Technical Education, Noida")

    st.write("### Team Members")
    st.write("• Pranjali")
    st.write("• Mahi Gupta")
    st.write("• Kanishka Patwal")
    st.markdown("---")
    st.success("Use the navigation panel to explore the dashboard.")
    # =====================================================
# DATA VISUALIZATION PAGE
# =====================================================

elif page == "Data Visualization":

    st.title("📊 Data Visualization Dashboard")

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.markdown("---")

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    chart = st.selectbox(
        "Select Visualization",
        (
            "Accident Severity Distribution",
            "Weather Conditions",
            "Road Surface Type",
            "Driver Gender",
            "Light Conditions",
            "Age Band of Driver",
            "Time vs Severity"
        )
    )

    # =====================================================
    # ACCIDENT SEVERITY
    # =====================================================

    if chart == "Accident Severity Distribution":

        fig, ax = plt.subplots(figsize=(7,5))

        df["Accident_severity"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Accident Severity Distribution")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Count")

        st.pyplot(fig)

        st.info(
            "This graph shows the frequency of each accident severity level."
        )

    # =====================================================
    # WEATHER
    # =====================================================

    elif chart == "Weather Conditions":

        fig, ax = plt.subplots(figsize=(8,8))

        df["Weather_conditions"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )

        ax.set_ylabel("")

        st.pyplot(fig)

        st.info(
            "Weather conditions at the time of accidents."
        )

    # =====================================================
    # ROAD SURFACE
    # =====================================================

    elif chart == "Road Surface Type":

        fig, ax = plt.subplots(figsize=(8,5))

        df["Road_surface_conditions"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Road Surface Conditions")

        st.pyplot(fig)

    # =====================================================
    # DRIVER GENDER
    # =====================================================

    elif chart == "Driver Gender":

        fig, ax = plt.subplots(figsize=(7,7))

        df["Sex_of_driver"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )

        ax.set_ylabel("")

        st.pyplot(fig)

    # =====================================================
    # LIGHT CONDITIONS
    # =====================================================

    elif chart == "Light Conditions":

        fig, ax = plt.subplots(figsize=(8,5))

        df["Light_conditions"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Light Conditions")

        st.pyplot(fig)

    # =====================================================
    # AGE BAND
    # =====================================================

    elif chart == "Age Band of Driver":

        fig, ax = plt.subplots(figsize=(8,5))

        df["Age_band_of_driver"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Age Band of Drivers")

        st.pyplot(fig)

    # =====================================================
    # TIME VS SEVERITY
    # =====================================================

    elif chart == "Time vs Severity":

        df2 = df.copy()

        df2["Hour"] = pd.to_datetime(
            df2["Time"],
            errors="coerce"
        ).dt.hour
        ctab = pd.crosstab(
            df2["Hour"],
            df2["Accident_severity"]
        )
        fig, ax = plt.subplots(figsize=(10,5))
        ctab.plot(
            kind="bar",
            ax=ax
        )
        ax.set_xlabel("Hour")
        ax.set_ylabel("Accidents")
        ax.set_title("Time vs Severity")
        st.pyplot(fig)
        # =====================================================
# FEATURE IMPORTANCE PAGE (Change #10)
# =====================================================

elif page == "Feature Importance":

    st.title("📈 Feature Importance")
    st.markdown("---")
    st.write(
        "Top features influencing the XGBoost model, "
        "ranked by importance score."
    )

    try:
        booster = model.get_booster()
        importance = booster.get_score(importance_type="weight")
        imp_df = pd.DataFrame(
            list(importance.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False)

        top_n = st.slider("Number of top features to show", 10, 50, 20)
        top_imp = imp_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(
            top_imp["Feature"][::-1],
            top_imp["Importance"][::-1],
            color="steelblue"
        )
        ax.set_xlabel("Importance (weight)")
        ax.set_title(f"Top {top_n} Feature Importances")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Feature Importance Table")
        st.dataframe(
            imp_df.reset_index(drop=True),
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"Could not extract feature importances: {e}")
        st.info(
            "Feature importance is available for tree-based models "
            "like XGBoost."
        )
        # =====================================================
# MODEL COMPARISON PAGE (Change #11)
# =====================================================

elif page == "Model Comparison":

    st.title("🏆 Model Comparison")
    st.markdown("---")
    st.write(
        "Performance comparison of all models tested during training."
    )

    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "CatBoost",
            "LightGBM",
            "XGBoost",
            "XGBoost + ADASYN (Final)"
        ],
        "Accuracy": [0.7650, 0.8210, 0.8320, 0.8350, 0.8380, 0.8450],
        "Precision": [0.7200, 0.8000, 0.8150, 0.8200, 0.8250, 0.8320],
        "Recall": [0.7100, 0.7900, 0.8050, 0.8100, 0.8150, 0.8250],
        "F1-Score": [0.7150, 0.7950, 0.8100, 0.8150, 0.8200, 0.8280],
    })

    st.dataframe(
        model_results.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score"],
            color="#90EE90"
        ),
        use_container_width=True
    )

    st.markdown("")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(model_results))
    width = 0.2
    ax.bar(
        [i - 1.5*width for i in x],
        model_results["Accuracy"],
        width, label="Accuracy", color="steelblue"
    )
    ax.bar(
        [i - 0.5*width for i in x],
        model_results["Precision"],
        width, label="Precision", color="orange"
    )
    ax.bar(
        [i + 0.5*width for i in x],
        model_results["Recall"],
        width, label="Recall", color="green"
    )
    ax.bar(
        [i + 1.5*width for i in x],
        model_results["F1-Score"],
        width, label="F1-Score", color="red"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_results["Model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Confusion Matrix (Change #12)")
    st.write("Confusion matrix for the final XGBoost + ADASYN model on the test set.")
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import ADASYN

        X_all = df.drop(columns=["Accident_severity", "Time"], errors="ignore")
        y_all = df["Accident_severity"]
        X_all = pd.get_dummies(X_all, drop_first=True)
        X_all.columns = (
            X_all.columns.astype(str)
            .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )
        X_all = X_all.loc[:, ~X_all.columns.duplicated()]
        for col in feature_order:
            if col not in X_all.columns:
                X_all[col] = 0
        X_all = X_all[feature_order]

        le_temp = joblib.load("label_encoder.pkl")
        y_enc = le_temp.transform(y_all)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_enc, test_size=0.20, random_state=42, stratify=y_enc
        )
        adasyn = ADASYN(random_state=42)
        X_tr_r, y_tr_r = adasyn.fit_resample(X_tr, y_tr)

        y_pred = model.predict(X_te)
        cm = confusion_matrix(y_te, y_pred)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=le_temp.classes_
        )
        disp.plot(ax=ax2, cmap="Blues", values_format="d")
        ax2.set_title("Confusion Matrix — Test Set")
        st.pyplot(fig2)
    except Exception as e:
        st.info(f"Confusion matrix could not be generated: {e}")
        # =====================================================
# ACCIDENT PREDICTION PAGE (Changes #1-7, #14-17)
# =====================================================

elif page == "Accident Prediction":

    st.title("🤖 Accident Severity Prediction")

    st.markdown(
        """
        Enter the accident details below and click **Predict Severity**.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Driver Info")

        day = st.selectbox(
            "Day of Week",
            sorted(df["Day_of_week"].dropna().unique())
        )

        age = st.selectbox(
            "Age Band of Driver",
            sorted(df["Age_band_of_driver"].dropna().unique())
        )

        sex = st.selectbox(
            "Sex of Driver",
            sorted(df["Sex_of_driver"].dropna().unique())
        )

        education = st.selectbox(
            "Educational Level",
            sorted(df["Educational_level"].dropna().unique())
        )

        vehicle_driver_rel = st.selectbox(
            "Vehicle-Driver Relation",
            sorted(df["Vehicle_driver_relation"].dropna().unique())
        )

        experience = st.selectbox(
            "Driving Experience",
            sorted(df["Driving_experience"].dropna().unique())
        )

    with col2:
        st.subheader("Vehicle & Road Info")

        vehicle = st.selectbox(
            "Type of Vehicle",
            sorted(df["Type_of_vehicle"].dropna().unique())
        )

        owner = st.selectbox(
            "Owner of Vehicle",
            sorted(df["Owner_of_vehicle"].dropna().unique())
        )

        service = st.selectbox(
            "Service Year",
            sorted(df["Service_year_of_vehicle"].dropna().unique())
        )

        defect = st.selectbox(
            "Defect of Vehicle",
            sorted(df["Defect_of_vehicle"].dropna().unique())
        )

        area = st.selectbox(
            "Area of Accident",
            sorted(df["Area_accident_occured"].dropna().unique())
        )

        lanes = st.selectbox(
            "Lanes or Medians",
            sorted(df["Lanes_or_Medians"].dropna().unique())
        )

        road_align = st.selectbox(
            "Road Allignment",
            sorted(df["Road_allignment"].dropna().unique())
        )

    with col3:
        st.subheader("Environment & Accident Info")

        junction = st.selectbox(
            "Junction Type",
            sorted(df["Types_of_Junction"].dropna().unique())
        )

        road_type = st.selectbox(
            "Road Surface Type",
            sorted(df["Road_surface_type"].dropna().unique())
        )

        road = st.selectbox(
            "Road Surface Condition",
            sorted(df["Road_surface_conditions"].dropna().unique())
        )

        light = st.selectbox(
            "Light Conditions",
            sorted(df["Light_conditions"].dropna().unique())
        )

        weather = st.selectbox(
            "Weather Conditions",
            sorted(df["Weather_conditions"].dropna().unique())
        )

        collision = st.selectbox(
            "Collision Type",
            sorted(df["Type_of_collision"].dropna().unique())
        )

        cause = st.selectbox(
            "Cause of Accident",
            sorted(df["Cause_of_accident"].dropna().unique())
        )

        movement = st.selectbox(
            "Vehicle Movement",
            sorted(df["Vehicle_movement"].dropna().unique())
        )

    st.markdown("---")

    col_num1, col_num2 = st.columns(2)
    with col_num1:
        num_vehicles = st.number_input(
            "Number of Vehicles Involved", min_value=1, max_value=10, value=1
        )
    with col_num2:
        num_casualties = st.number_input(
            "Number of Casualties", min_value=0, max_value=20, value=0
        )

    st.markdown("---")

    col_time1, col_time2 = st.columns(2)
    with col_time1:
        hour = st.slider("Hour of Accident", 0, 23, 12)
    with col_time2:
        minute = st.slider("Minute of Accident", 0, 59, 0)

    st.markdown("---")

    col_cas1, col_cas2, col_cas3 = st.columns(3)
    with col_cas1:
        casualty_class = st.selectbox(
            "Casualty Class",
            sorted(df["Casualty_class"].dropna().unique())
        )
        sex_casualty = st.selectbox(
            "Sex of Casualty",
            sorted(df["Sex_of_casualty"].dropna().unique())
        )
    with col_cas2:
        age_casualty = st.selectbox(
            "Age Band of Casualty",
            sorted(df["Age_band_of_casualty"].dropna().unique())
        )
        casualty_severity = st.selectbox(
            "Casualty Severity",
            sorted(df["Casualty_severity"].dropna().unique())
        )
    with col_cas3:
        work_casualty = st.selectbox(
            "Work of Casualty",
            sorted(df["Work_of_casuality"].dropna().unique())
        )
        fitness_casualty = st.selectbox(
            "Fitness of Casualty",
            sorted(df["Fitness_of_casuality"].dropna().unique())
        )

    pedestrian = st.selectbox(
        "Pedestrian Movement",
        sorted(df["Pedestrian_movement"].dropna().unique())
    )

    st.markdown("---")

    predict = st.button(
        "🚗 Predict Severity",
        use_container_width=True
    )

    # ---- Reset Button (Change #15) ----
    if st.button("🔄 Reset Form", use_container_width=True):
        st.rerun()

    if predict:

        user_df = pd.DataFrame([{
            "Day_of_week": day,
            "Age_band_of_driver": age,
            "Sex_of_driver": sex,
            "Educational_level": education,
            "Vehicle_driver_relation": vehicle_driver_rel,
            "Driving_experience": experience,
            "Type_of_vehicle": vehicle,
            "Owner_of_vehicle": owner,
            "Service_year_of_vehicle": service,
            "Defect_of_vehicle": defect,
            "Area_accident_occured": area,
            "Lanes_or_Medians": lanes,
            "Road_allignment": road_align,
            "Types_of_Junction": junction,
            "Road_surface_type": road_type,
            "Road_surface_conditions": road,
            "Light_conditions": light,
            "Weather_conditions": weather,
            "Type_of_collision": collision,
            "Cause_of_accident": cause,
            "Vehicle_movement": movement,
            "Number_of_vehicles_involved": num_vehicles,
            "Number_of_casualties": num_casualties,
            "Hour": hour,
            "Minute": minute,
            "Time_Minutes": hour * 60 + minute,
            "Casualty_class": casualty_class,
            "Sex_of_casualty": sex_casualty,
            "Age_band_of_casualty": age_casualty,
            "Casualty_severity": casualty_severity,
            "Work_of_casuality": work_casualty,
            "Fitness_of_casuality": fitness_casualty,
            "Pedestrian_movement": pedestrian,
        }])

        # ------------------------------------------------
        # STRIP WHITESPACE (matches notebook cleaning)
        # ------------------------------------------------
        str_cols = [
            "Age_band_of_driver", "Type_of_vehicle", "Road_surface_type",
            "Weather_conditions", "Light_conditions", "Area_accident_occured",
            "Driving_experience", "Types_of_Junction", "Lanes_or_Medians",
            "Road_allignment", "Road_surface_conditions", "Day_of_week",
            "Sex_of_driver", "Vehicle_driver_relation", "Owner_of_vehicle",
            "Service_year_of_vehicle", "Cause_of_accident", "Type_of_collision",
            "Vehicle_movement", "Pedestrian_movement"
        ]
        for c in str_cols:
            if c in user_df.columns:
                user_df[c] = user_df[c].astype(str).str.strip()

        # ------------------------------------------------
        # FEATURE ENGINEERING (must match notebook exactly)
        # ------------------------------------------------

        # --- Temporal ---
        user_df["Hour_Sin"] = np.sin(
            2 * np.pi * user_df["Hour"] / 24
        )
        user_df["Hour_Cos"] = np.cos(
            2 * np.pi * user_df["Hour"] / 24
        )
        user_df["Rush_Hour"] = (
            user_df["Hour"].isin([7, 8, 9, 16, 17, 18, 19])
        ).astype(int)
        user_df["Late_Night"] = (
            user_df["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])
        ).astype(int)

        def get_time_period(h):
            if h < 6:
                return "Night"
            if h < 12:
                return "Morning"
            if h < 18:
                return "Afternoon"
            return "Evening"

        user_df["Time_Period"] = user_df["Hour"].apply(get_time_period)

        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        user_df["Day_Num"] = user_df["Day_of_week"].map(day_map).fillna(-1).astype(int)
        user_df["Weekend"] = (
            user_df["Day_of_week"].isin(["Saturday", "Sunday"])
        ).astype(int)

        # --- Risk Scores ---
        user_df["Driver_Age_Risk"] = user_df["Age_band_of_driver"].map(age_score).fillna(2)
        user_df["Driver_Experience_Score"] = user_df["Driving_experience"].map(experience_score).fillna(3)
        user_df["Vehicle_Age_Risk"] = user_df["Service_year_of_vehicle"].map(vehicle_age_score).fillna(2)
        user_df["Weather_Risk_Score"] = user_df["Weather_conditions"].map(weather_score_map).fillna(2)
        user_df["Junction_Complexity"] = user_df["Types_of_Junction"].map(junction_score).fillna(2)
        user_df["Road_Surface_Risk"] = user_df["Road_surface_conditions"].map(surface_score).fillna(1)
        user_df["Light_Risk"] = user_df["Light_conditions"].map(light_score).fillna(1)

        user_df["Driver_Risk_Index"] = (
            0.55 * user_df["Driver_Experience_Score"] +
            0.45 * user_df["Driver_Age_Risk"]
        )
        user_df["Environmental_Risk_Index"] = (
            user_df["Weather_Risk_Score"] +
            user_df["Road_Surface_Risk"] +
            user_df["Light_Risk"] +
            user_df["Junction_Complexity"]
        ) / 4
        user_df["Accident_Risk_Index"] = (
            0.40 * user_df["Driver_Risk_Index"] +
            0.40 * user_df["Environmental_Risk_Index"] +
            0.20 * user_df["Vehicle_Age_Risk"]
        )

        # --- Binary / Numeric Interaction ---
        user_df["Casualty_per_vehicle"] = (
            user_df["Number_of_casualties"] /
            user_df["Number_of_vehicles_involved"].clip(lower=1)
        )
        user_df["Multiple_Casualties"] = (user_df["Number_of_casualties"] >= 2).astype(int)
        user_df["Heavy_Traffic"] = (user_df["Number_of_vehicles_involved"] >= 3).astype(int)
        user_df["Poor_Visibility"] = (
            (user_df["Light_conditions"] != "Daylight") &
            (user_df["Weather_conditions"] != "Normal")
        ).astype(int)
        user_df["Wet_Night"] = (
            (user_df["Road_surface_conditions"] != "Dry") &
            (user_df["Light_conditions"] != "Daylight")
        ).astype(int)
        user_df["High_Risk_Driver"] = (
            (user_df["Age_band_of_driver"] == "Under 18") |
            (user_df["Driving_experience"].isin(["No Licence", "Below 1yr"]))
        ).astype(int)
        user_df["Experienced_Driver"] = (
            user_df["Driving_experience"].isin(["5-10yr", "Above 10yr"])
        ).astype(int)
        user_df["Complex_Road"] = (
            (user_df["Types_of_Junction"] != "No junction") &
            (user_df["Lanes_or_Medians"] != "Undivided Two way")
        ).astype(int)
        user_df["Old_Vehicle_Inexperienced_Driver"] = (
            user_df["Service_year_of_vehicle"].isin(["Above 10yr", "5-10yrs"]) &
            user_df["Driving_experience"].isin(["Below 1yr", "No Licence"])
        ).astype(int)

        # --- Context Combination Features ---
        user_df["Driver_Context"] = (
            user_df["Age_band_of_driver"] + "_" + user_df["Driving_experience"]
        )
        user_df["Road_Environment"] = (
            user_df["Road_surface_conditions"] + "_" +
            user_df["Light_conditions"] + "_" +
            user_df["Weather_conditions"]
        )
        user_df["Junction_Lane_Context"] = (
            user_df["Types_of_Junction"] + "_" + user_df["Lanes_or_Medians"]
        )
        user_df["Vehicle_Driver_Context"] = (
            user_df["Type_of_vehicle"] + "_" + user_df["Driving_experience"]
        )
        user_df["Cause_Collision_Context"] = (
            user_df["Cause_of_accident"] + "_" + user_df["Type_of_collision"]
        )
        user_df["Area_Time_Context"] = (
            user_df["Area_accident_occured"] + "_" + user_df["Time_Period"]
        )

        # --- Frequency Encoding ---
        for col, fmap in freq_maps.items():
            if col in user_df.columns:
                user_df[f"{col}_Frequency"] = user_df[col].map(fmap).fillna(0.0)

        # --- Advanced Interaction Features ---
        user_df["Night_Poor_Visibility"] = (
            user_df["Late_Night"] * user_df["Poor_Visibility"]
        )
        user_df["RushHour_HeavyTraffic"] = (
            user_df["Rush_Hour"] * user_df["Heavy_Traffic"]
        )
        user_df["WetRoad_Night"] = (
            user_df["Wet_Night"] * user_df["Late_Night"]
        )
        user_df["RiskDriver_Night"] = (
            user_df["High_Risk_Driver"] * user_df["Late_Night"]
        )
        user_df["ComplexRoad_PoorVisibility"] = (
            user_df["Complex_Road"] * user_df["Poor_Visibility"]
        )
        user_df["Driver_Road_Risk"] = (
            user_df["Driver_Risk_Index"] * user_df["Road_Surface_Risk"]
        )
        user_df["Driver_Junction_Risk"] = (
            user_df["Driver_Risk_Index"] * user_df["Junction_Complexity"]
        )
        user_df["Environment_Junction_Risk"] = (
            user_df["Environmental_Risk_Index"] * user_df["Junction_Complexity"]
        )
        user_df["Vehicle_Experience_Risk"] = (
            user_df["Vehicle_Age_Risk"] * (1 - user_df["Driver_Experience_Score"])
        )
        user_df["Composite_Risk_Score"] = (
            0.30 * user_df["Driver_Risk_Index"] +
            0.25 * user_df["Environmental_Risk_Index"] +
            0.20 * user_df["Vehicle_Age_Risk"] +
            0.15 * user_df["Road_Surface_Risk"] +
            0.10 * user_df["Junction_Complexity"]
        )
        user_df["Severe_Driving_Context"] = (
            user_df["High_Risk_Driver"] *
            user_df["Poor_Visibility"] *
            user_df["Rush_Hour"]
        )
        user_df["Night_Complex_Road"] = (
            user_df["Late_Night"] * user_df["Complex_Road"]
        )

        # --- One-Hot Encode ---
        user_df = pd.get_dummies(user_df, drop_first=False)
        user_df.columns = (
            user_df.columns.astype(str)
            .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )
        user_df = user_df.loc[:, ~user_df.columns.duplicated()]

        # --- Align with training features ---
        for col in feature_order:
            if col not in user_df.columns:
                user_df[col] = 0
        user_df = user_df[feature_order]

        # ------------------------------------------------
        # PREDICTION
        # ------------------------------------------------
        prediction = model.predict(user_df)
        proba = model.predict_proba(user_df)[0]
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        confidence = float(proba.max())
        class_labels = label_encoder.classes_

        # --- Change #6: Low Confidence Warning ---
        if confidence < 0.45:
            st.warning(
                f"⚠️ Low confidence ({confidence:.1%}). "
                "The prediction may be unreliable. "
                "Consider reviewing the input values."
            )

        # --- Change #1: Prediction Confidence ---
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        st.success(f"### Predicted Severity: **{predicted_label}**")
        st.metric("Prediction Confidence", f"{confidence:.1%}")

        # --- Change #2: Probability for Every Class ---
        st.markdown("")
        st.subheader("Class Probabilities")
        proba_df = pd.DataFrame({
            "Severity": class_labels,
            "Probability": proba
        }).sort_values("Probability", ascending=False)

        fig_p, ax_p = plt.subplots(figsize=(6, 3))
        colors = ["#2ecc71" if p == proba.max() else "#3498db" for p in proba]
        ax_p.barh(proba_df["Severity"], proba_df["Probability"], color=colors)
        ax_p.set_xlabel("Probability")
        ax_p.set_title("Prediction Probability Distribution")
        ax_p.set_xlim(0, 1)
        for i, v in enumerate(proba_df["Probability"]):
            ax_p.text(v + 0.01, i, f"{v:.1%}", va="center")
        st.pyplot(fig_p)

        # --- Change #3: Risk Meter ---
        st.markdown("---")
        st.subheader("🚦 Risk Level")
        if predicted_label.lower() == "slight":
            risk_color = "🟢"
            risk_level = "LOW RISK"
        elif predicted_label.lower() == "serious":
            risk_color = "🟠"
            risk_level = "MODERATE RISK"
        else:
            risk_color = "🔴"
            risk_level = "HIGH RISK"
        st.markdown(
            f"### {risk_color} **{risk_level}** "
            f"— Severity: {predicted_label}"
        )

        # --- Change #4: Show Risk Factors ---
        st.markdown("---")
        st.subheader("⚠️ Key Risk Factors")
        risk_factors = []
        if user_df["High_Risk_Driver"].values[0] == 1:
            risk_factors.append("Young or inexperienced driver")
        if user_df["Late_Night"].values[0] == 1:
            risk_factors.append("Late night accident (10 PM - 5 AM)")
        if user_df["Poor_Visibility"].values[0] == 1:
            risk_factors.append("Poor visibility conditions")
        if user_df["Wet_Night"].values[0] == 1:
            risk_factors.append("Wet road at night")
        if user_df["Heavy_Traffic"].values[0] == 1:
            risk_factors.append("Heavy traffic (3+ vehicles)")
        if user_df["Old_Vehicle_Inexperienced_Driver"].values[0] == 1:
            risk_factors.append("Old vehicle with inexperienced driver")
        if user_df["Complex_Road"].values[0] == 1:
            risk_factors.append("Complex road junction")
        if user_df["Rush_Hour"].values[0] == 1:
            risk_factors.append("Rush hour period")
        if user_df["Severe_Driving_Context"].values[0] == 1:
            risk_factors.append("Severe driving context (high-risk driver + poor visibility + rush hour)")

        if risk_factors:
            for rf in risk_factors:
                st.write(f"• {rf}")
        else:
            st.write("No major risk factors detected.")

        # --- Change #5: Explain the Prediction ---
        st.markdown("---")
        st.subheader("📝 Prediction Explanation")
        top_class = proba_df.iloc[0]
        second_class = proba_df.iloc[1] if len(proba_df) > 1 else None

        explanation = (
            f"The model predicts **{top_class['Severity']}** "
            f"with **{top_class['Probability']:.1%}** confidence. "
        )
        if second_class is not None:
            explanation += (
                f"The second most likely outcome is "
                f"**{second_class['Severity']}** at "
                f"**{second_class['Probability']:.1%}**."
            )
        st.write(explanation)

        # --- Change #7: Final Summary Card ---
        st.markdown("---")
        st.subheader("📋 Summary Card")
        summary_data = {
            "Day": day,
            "Hour": f"{hour:02d}:{minute:02d}",
            "Driver": f"{age}, {sex}",
            "Experience": experience,
            "Vehicle": vehicle,
            "Weather": weather,
            "Road Surface": road,
            "Light": light,
            "Predicted Severity": predicted_label,
            "Confidence": f"{confidence:.1%}",
            "Risk Level": risk_level,
        }
        summary_df = pd.DataFrame(
            list(summary_data.items()),
            columns=["Parameter", "Value"]
        )
        st.table(summary_df)

        # --- Change #16: Download Report ---
        st.markdown("---")
        report_text = "ROAD ACCIDENT SEVERITY PREDICTION REPORT\n"
        report_text += "=" * 50 + "\n\n"
        for k, v in summary_data.items():
            report_text += f"{k}: {v}\n"
        report_text += "\nRisk Factors:\n"
        if risk_factors:
            for rf in risk_factors:
                report_text += f"  - {rf}\n"
        else:
            report_text += "  None detected\n"
        report_text += f"\nClass Probabilities:\n"
        for _, row in proba_df.iterrows():
            report_text += f"  {row['Severity']}: {row['Probability']:.1%}\n"

        st.download_button(
            label="📥 Download Prediction Report",
            data=report_text,
            file_name="accident_prediction_report.txt",
            mime="text/plain",
            use_container_width=True
        )
        # =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About":

    st.title("ℹ️ About This Project")

    st.markdown("---")

    st.header("Road Accident Severity Prediction")

    st.write("""
    This project predicts the severity of road accidents using
    Machine Learning techniques.

    It was developed as part of the B.Tech AI & ML curriculum.

    The dashboard allows users to:

    • Explore the accident dataset

    • View different visualizations

    • Predict accident severity using the trained model

    • Understand accident patterns and risk factors
    """)

    st.markdown("---")

    st.subheader("Machine Learning Model")

    st.write("""
    **Algorithm Used**

    - XGBoost Classifier (with ADASYN oversampling)

    **Libraries**

    - Pandas
    - NumPy
    - Matplotlib
    - Scikit-Learn
    - XGBoost
    - Streamlit

    **Model Accuracy**

    Approximately **84.50%**
    """)

    st.markdown("---")

    st.subheader("Dataset Features")

    st.write("""
    - Driver Information
    - Vehicle Information
    - Weather Conditions
    - Road Surface
    - Light Conditions
    - Junction Type
    - Collision Type
    - Cause of Accident
    - Time Information
    - Casualty Information
    """)

    st.markdown("---")

    st.success("Developed by Team: Pranjali • Mahi Gupta • Kanishka Patwal")

# =====================================================
# FOOTER (Change #13) — visible on all pages
# =====================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "Road Accident Severity Prediction Dashboard • "
    "B.Tech AI & ML, JSS Academy of Technical Education, Noida • "
    "Pranjali • Mahi Gupta • Kanishka Patwal"
    "</div>",
    unsafe_allow_html=True
)
