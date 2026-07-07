import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Road.csv")
# Graph 1: Accident severity
severity_counts=df["Accident_severity"].value_counts()
plt.figure(figsize=(6,4))
severity_counts.plot(kind="bar")
plt.title("Accident severity distribution")
plt.xlabel("Severity")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Graph 2: Time vs Severity
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
time_severity = pd.crosstab(df["Hour"], df["Accident_severity"])
time_severity.plot(kind="bar", figsize=(10,5))
plt.title("Time vs Accident Severity")
plt.show()


# Graph 3: Weather Conditions Distribution
weather_counts = df["Weather_conditions"].value_counts()
weather_counts.plot(kind="bar",figsize=(10,6),color="skyblue",edgecolor="black")
plt.title("Distribution of Accidents by Weather Condition", fontsize=15, fontweight="bold")
plt.xlabel("Weather Conditions")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Graph 4: Day of Week Distribution
day_counts = df["Day_of_week"].value_counts()
days = [ "Monday", "Tuesday", "Wednesday","Thursday", "Friday", "Saturday", "Sunday"]
day_counts = day_counts.reindex(days)
plt.figure(figsize=(10,5))
plt.plot(day_counts.index, day_counts.values, marker="o", linewidth=2)
plt.title("Accidents by Day of Week", fontsize=15, fontweight="bold")
plt.xlabel("Day of Week")
plt.ylabel("Number of Accidents")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Graph 5: Light Conditions Distribution
light_counts = df["Light_conditions"].value_counts()
plt.figure(figsize=(8,8))
wedges, texts, autotexts = plt.pie( light_counts, autopct="%1.1f%%",  startangle=90,  pctdistance=0.75, wedgeprops={"edgecolor": "black"})
plt.title("Distribution of Accidents by Light Conditions", fontsize=15, fontweight="bold")
plt.legend(wedges, light_counts.index, title="Light Conditions", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()


# Graph 6: Road Surface Conditions Distribution
surface_counts = df["Road_surface_conditions"].value_counts()
plt.figure(figsize=(10,6))
surface_counts.plot( kind="barh", color="teal", edgecolor="black")
plt.title("Distribution of Accidents by Road Surface Conditions",fontsize=15, fontweight="bold")
plt.xlabel("Number of Accidents")
plt.ylabel("Road Surface Conditions")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Graph 7: Vehicle Type Distribution
vehicle_counts = df["Type_of_vehicle"].value_counts()
plt.figure(figsize=(10,6))
vehicle_counts.plot( kind="barh",color="orange", edgecolor="black")
plt.title("Distribution of Accidents by Vehicle Type",fontsize=15,fontweight="bold")
plt.xlabel("Number of Accidents")
plt.ylabel("Vehicle Type")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Graph 8: Driver Age Band Distribution
age_counts = df["Age_band_of_driver"].value_counts()
plt.figure(figsize=(8,8))
wedges, texts, autotexts = plt.pie( age_counts, labels=age_counts.index, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "black"})
plt.title("Distribution of Accidents by Driver Age Group",fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()


# Graph 9: Top 10 Causes of Accident
cause_counts = (df["Cause_of_accident"] .value_counts().head(10).sort_values())
plt.figure(figsize=(10,6))
cause_counts.plot( kind="barh", color="crimson",edgecolor="black")
plt.title("Top 10 Causes of Accidents", fontsize=15, fontweight="bold")
plt.xlabel("Number of Accidents")
plt.ylabel("Cause of Accident")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Graph 10: Driver Gender Distribution
gender_counts = df["Sex_of_driver"].value_counts()
plt.figure(figsize=(8,8))
plt.pie(
    gender_counts,
    labels=gender_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    explode=[0.05]*len(gender_counts),  
    wedgeprops={"edgecolor":"black"})
plt.title("Distribution of Drivers by Gender", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()