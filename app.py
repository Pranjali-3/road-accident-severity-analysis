import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

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
# SIDEBAR
# =====================================================

st.sidebar.title("🚗 Road Accident Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Visualization",
        "🚗 Accident Prediction",
        "ℹ About"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.title("🚗 Road Accident Severity Prediction System")

    st.markdown("""
### Department of Computer Science & Engineering (AI & ML)

**University:** JSS Academy of Technical Education, Noida

---

### Project Description

This project analyzes road accident data to understand the factors affecting accident severity.

Machine Learning techniques are used to classify accidents into different severity levels based on:

- Driver Information
- Weather Conditions
- Road Surface
- Vehicle Details
- Environmental Factors

---

### Best Performing Model

✅ Enhanced XGBoost

### Accuracy

**85.23%**

---

### Team Members

- Pranjali
- Mahi Gupta
- Kanishka Patwal
""")

# =====================================================
# DATA VISUALIZATION
# =====================================================

elif page == "📊 Data Visualization":

    st.title("📊 Road Accident Analysis Dashboard")

    st.info(
        "Use the filters on the left to explore accident patterns."
    )

    # ---------------- Sidebar Filters ----------------

    st.sidebar.header("🔍 Filters")

    gender = st.sidebar.selectbox(
        "Driver Gender",
        ["All"] + sorted(df["Sex_of_driver"].dropna().unique())
    )

    severity = st.sidebar.selectbox(
        "Accident Severity",
        ["All"] + sorted(df["Accident_severity"].dropna().unique())
    )

    weather = st.sidebar.selectbox(
        "Weather Condition",
        ["All"] + sorted(df["Weather_conditions"].dropna().unique())
    )

    vehicle = st.sidebar.selectbox(
        "Vehicle Type",
        ["All"] + sorted(df["Type_of_vehicle"].dropna().unique())
    )

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    available_days = [
        d for d in days
        if d in df["Day_of_week"].unique()
    ]

    day = st.sidebar.selectbox(
        "Day of Week",
        ["All"] + available_days
    )

    # ---------------- Filtering ----------------

    filtered_df = df.copy()

    if gender != "All":
        filtered_df = filtered_df[
            filtered_df["Sex_of_driver"] == gender
        ]

    if severity != "All":
        filtered_df = filtered_df[
            filtered_df["Accident_severity"] == severity
        ]

    if weather != "All":
        filtered_df = filtered_df[
            filtered_df["Weather_conditions"] == weather
        ]

    if vehicle != "All":
        filtered_df = filtered_df[
            filtered_df["Type_of_vehicle"] == vehicle
        ]

    if day != "All":
        filtered_df = filtered_df[
            filtered_df["Day_of_week"] == day
        ]

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        st.stop()

    # =====================================================
    # KPI CARDS
    # =====================================================

    st.subheader("📈 Dashboard Summary")

    fatal = len(filtered_df[
        filtered_df["Accident_severity"] == "Fatal injury"
    ])

    serious = len(filtered_df[
        filtered_df["Accident_severity"] == "Serious Injury"
    ])

    slight = len(filtered_df[
        filtered_df["Accident_severity"] == "Slight Injury"
    ])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🚗 Total Accidents", len(filtered_df))
    c2.metric("💀 Fatal", fatal)
    c3.metric("⚠ Serious", serious)
    c4.metric("🩹 Slight", slight)

    st.divider()

    # =====================================================
    # GRAPH ROW 1
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Accident Severity Distribution")

        severity_count = filtered_df[
            "Accident_severity"
        ].value_counts()

        fig, ax = plt.subplots(figsize=(6,4))

        severity_count.plot(
            kind="bar",
            color=["red","orange","green"],
            ax=ax
        )

        ax.set_xlabel("Severity")
        ax.set_ylabel("Accidents")
        ax.tick_params(axis='x', rotation=20)

        st.pyplot(fig)

    with col2:

        st.subheader("Driver Gender Distribution")

        gender_count = filtered_df[
            "Sex_of_driver"
        ].value_counts()

        fig, ax = plt.subplots(figsize=(6,6))

        ax.pie(
            gender_count,
            labels=gender_count.index,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor":"black"}
        )

        st.pyplot(fig)

    # =====================================================
    # GRAPH ROW 2
    # =====================================================

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Weather Condition Distribution")

        weather_count = filtered_df[
            "Weather_conditions"
        ].value_counts()

        fig, ax = plt.subplots(figsize=(6,4))

        weather_count.plot(
            kind="bar",
            color="skyblue",
            ax=ax
        )

        ax.tick_params(axis='x', rotation=45)

        st.pyplot(fig)

    with col4:

        st.subheader("Accidents by Day")

        day_count = filtered_df[
            "Day_of_week"
        ].value_counts().reindex(days)

        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(
            day_count.index,
            day_count.values,
            marker="o",
            linewidth=3
        )

        ax.grid(True)

        st.pyplot(fig)

    # =====================================================
    # GRAPH ROW 3
    # =====================================================

    col5, col6 = st.columns(2)

    with col5:

        st.subheader("Top 10 Vehicle Types")

        vehicle_count = filtered_df[
            "Type_of_vehicle"
        ].value_counts().head(10).sort_values()

        fig, ax = plt.subplots(figsize=(6,4))

        vehicle_count.plot(
            kind="barh",
            color="teal",
            ax=ax
        )

        st.pyplot(fig)

    with col6:

        st.subheader("Top 10 Causes of Accidents")

        cause_count = filtered_df[
            "Cause_of_accident"
        ].value_counts().head(10).sort_values()

        fig, ax = plt.subplots(figsize=(6,4))

        cause_count.plot(
            kind="barh",
            color="tomato",
            ax=ax
        )

        st.pyplot(fig)

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🚗 Accident Prediction":

    st.title("🚗 Accident Severity Prediction")

    st.info(
        "Machine Learning prediction module will be integrated after model deployment."
    )

# =====================================================
# ABOUT
# =====================================================

elif page == "ℹ About":

    st.title("ℹ About This Project")

    st.markdown("""
### Road Accident Severity Prediction System

This project was developed to analyze road accident data and identify patterns affecting accident severity.

### Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost
- LightGBM

---

### Best Model
Enhanced XGBoost
### Accuracy
**85.23%**
""")