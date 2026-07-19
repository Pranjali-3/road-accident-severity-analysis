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
# SIDEBAR
# =====================================================
st.sidebar.title("🚗 Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Data Visualization",
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

        The final deployed model is the **Enhanced XGBoost Model**
        which achieved approximately **85.23% accuracy**.
        """
    )
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
# ACCIDENT PREDICTION PAGE
# =====================================================

elif page == "Accident Prediction":

    st.title("🤖 Accident Severity Prediction")

    st.markdown(
        """
        Enter the accident details below and click **Predict Severity**.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

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

        experience = st.selectbox(
            "Driving Experience",
            sorted(df["Driving_experience"].dropna().unique())
        )

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

    with col2:

        weather = st.selectbox(
            "Weather Conditions",
            sorted(df["Weather_conditions"].dropna().unique())
        )

        road = st.selectbox(
            "Road Surface",
            sorted(df["Road_surface_conditions"].dropna().unique())
        )

        light = st.selectbox(
            "Light Conditions",
            sorted(df["Light_conditions"].dropna().unique())
        )

        junction = st.selectbox(
            "Junction Type",
            sorted(df["Types_of_Junction"].dropna().unique())
        )

        collision = st.selectbox(
            "Collision Type",
            sorted(df["Type_of_collision"].dropna().unique())
        )

        cause = st.selectbox(
            "Cause of Accident",
            sorted(df["Cause_of_accident"].dropna().unique())
        )

        hour = st.slider(
            "Hour of Accident",
            0,
            23,
            12
        )

    st.markdown("---")

    predict = st.button(
        "🚗 Predict Severity",
        use_container_width=True
    )

    if predict:

        user_df = pd.DataFrame([{
            "Age_band_of_driver": age,
            "Sex_of_driver": sex,
            "Educational_level": education,
            "Driving_experience": experience,
            "Type_of_vehicle": vehicle,
            "Owner_of_vehicle": owner,
            "Service_year_of_vehicle": service,
            "Weather_conditions": weather,
            "Road_surface_conditions": road,
            "Light_conditions": light,
            "Types_of_Junction": junction,
            "Type_of_collision": collision,
            "Cause_of_accident": cause,
            "Hour": hour
        }])

        # -----------------------------
        # Feature Engineering
        # -----------------------------

        user_df["Hour_Sin"] = np.sin(
            2 * np.pi * user_df["Hour"] / 24
        )

        user_df["Hour_Cos"] = np.cos(
            2 * np.pi * user_df["Hour"] / 24
        )

        user_df["Rush_Hour"] = (
            (
                (user_df["Hour"] >= 7)
                &
                (user_df["Hour"] <= 10)
            )
            |
            (
                (user_df["Hour"] >= 16)
                &
                (user_df["Hour"] <= 19)
            )
        ).astype(int)

        # One-Hot Encoding

        user_df = pd.get_dummies(user_df)

        # Match training features

        for col in feature_order:

            if col not in user_df.columns:
                user_df[col] = 0

        user_df = user_df[feature_order]
        prediction = model.predict(user_df)
        prediction = label_encoder.inverse_transform(prediction)
        st.success(
            f"### Predicted Severity : {prediction[0]}"
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

    - XGBoost Classifier

    **Libraries**

    - Pandas
    - NumPy
    - Matplotlib
    - Scikit-Learn
    - XGBoost
    - Streamlit

    **Model Accuracy**

    Approximately **85.23%**
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
    """)

    st.markdown("---")

    st.success("Developed by Team: Pranjali • Mahi Gupta • Kanishka Patwal")