# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

# Show Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("Filter Options")
gender = st.sidebar.selectbox("Select Gender", options=df["Sex"].unique())
pclass = st.sidebar.selectbox("Select Passenger Class", options=sorted(df["Pclass"].unique()))

# Apply filters
filtered_df = df[(df["Sex"] == gender) & (df["Pclass"] == pclass)]

# Show filtered data
st.subheader("🎯 Filtered Data Preview")
st.write(filtered_df.head())

# ------------------------ VISUALIZATIONS ------------------------

# 1. Survival Count by Gender
st.subheader("1️⃣ Survival Count by Gender")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
ax1.set_title("Survival Count")
st.pyplot(fig1)

# 2. Age Distribution
st.subheader("2️⃣ Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=30, ax=ax2)
ax2.set_xlabel("Age")
ax2.set_ylabel("Number of Passengers")
ax2.set_title("Age Distribution of Passengers")
st.pyplot(fig2)

# 3. Survival Rate by Passenger Class
st.subheader("3️⃣ Survival Rate by Passenger Class")
fig3, ax3 = plt.subplots()
sns.barplot(data=filtered_df, x="Pclass", y="Survived", ax=ax3)
ax3.set_ylabel("Survival Rate")
ax3.set_title("Survival Rate vs Passenger Class")
st.pyplot(fig3)

# 4. Fare vs Age Scatter Plot
st.subheader("4️⃣ Fare vs Age Scatter Plot (Colored by Survival)")
fig4, ax4 = plt.subplots()
sns.scatterplot(data=filtered_df, x="Age", y="Fare", hue="Survived", palette="Set1", ax=ax4)
ax4.set_title("Fare vs Age")
st.pyplot(fig4)

# 5. Passenger Count by Embarked Port
st.subheader("5️⃣ Passenger Count by Embarked Port")
fig5, ax5 = plt.subplots()
sns.countplot(data=filtered_df, x="Embarked", hue="Survived", ax=ax5)
ax5.set_title("Embarked Port vs Survival")
st.pyplot(fig5)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
