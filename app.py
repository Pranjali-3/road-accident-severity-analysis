import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Road Accident Severity Dashboard",
    page_icon="🚗",
    layout="wide"
)
st.title("🚗 Road Accident Severity Dashboard")
st.markdown("### Data Visualization and Analysis of Road Accidents")
df = pd.read_csv("Road.csv")
st.sidebar.title("🔍 Filters")
gender = st.sidebar.selectbox(
    "Driver Gender",
    ["All"] + sorted(df["Sex_of_driver"].dropna().unique().tolist())
)
severity = st.sidebar.selectbox(
    "Accident Severity",
    ["All"] + sorted(df["Accident_severity"].dropna().unique().tolist())
)
weather = st.sidebar.selectbox(
    "Weather Condition",
    ["All"] + sorted(df["Weather_conditions"].dropna().unique().tolist())
)
days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
available_days = [d for d in days if d in df["Day_of_week"].unique()]
day = st.sidebar.selectbox(
    "Day of Week",
    ["All"] + available_days
)
vehicle = st.sidebar.selectbox(
    "Vehicle Type",
    ["All"] + sorted(df["Type_of_vehicle"].dropna().unique().tolist())
)
filtered_df = df.copy()
if gender != "All":
    filtered_df = filtered_df[filtered_df["Sex_of_driver"] == gender]
if severity != "All":
    filtered_df = filtered_df[filtered_df["Accident_severity"] == severity]
if weather != "All":
    filtered_df = filtered_df[filtered_df["Weather_conditions"] == weather]
if day != "All":
    filtered_df = filtered_df[filtered_df["Day_of_week"] == day]
if vehicle != "All":
    filtered_df = filtered_df[filtered_df["Type_of_vehicle"] == vehicle]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Accidents", len(filtered_df))
with col2:
    st.metric("Vehicle Types", filtered_df["Type_of_vehicle"].nunique())
with col3:
    st.metric("Weather Conditions", filtered_df["Weather_conditions"].nunique())
with col4:
    st.metric("Driver Genders", filtered_df["Sex_of_driver"].nunique())
col1, col2 = st.columns(2)
with col1:
    st.subheader("Accident Severity")
    severity_counts = filtered_df["Accident_severity"].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    severity_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Count")
    st.pyplot(fig)
with col2:
    st.subheader("Driver Gender")
    gender_counts = filtered_df["Sex_of_driver"].value_counts()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(
        gender_counts,
        labels=gender_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor":"black"}
    )
    st.pyplot(fig)
col3, col4 = st.columns(2)
with col3:
    st.subheader("Weather Conditions")
    weather_counts = filtered_df["Weather_conditions"].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    weather_counts.plot(kind="bar", ax=ax)
    st.pyplot(fig)
with col4:
    st.subheader("Accidents by Day")
    day_counts = filtered_df["Day_of_week"].value_counts().reindex(days)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(day_counts.index, day_counts.values, marker="o")
    st.pyplot(fig)
col5, col6 = st.columns(2)
with col5:
    st.subheader("Vehicle Types")
    vehicle_counts = filtered_df["Type_of_vehicle"].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    vehicle_counts.plot(kind="barh", ax=ax)
    st.pyplot(fig)
with col6:
    st.subheader("Top 10 Accident Causes")
    cause_counts = filtered_df["Cause_of_accident"].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(6,4))
    cause_counts.plot(kind="barh", ax=ax)
    st.pyplot(fig)