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

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
gender = st.sidebar.selectbox("Select Gender", options=df["Sex"].unique())
pclass = st.sidebar.selectbox("Select Passenger Class", options=sorted(df["Pclass"].unique()))

# Apply Filters
filtered_df = df[(df["Sex"] == gender) & (df["Pclass"] == pclass)]

# Show Filtered Data
st.subheader("🎯 Filtered Data Preview")
st.write(filtered_df.head())

# ------------------------ VISUALIZATIONS ------------------------

# 🔹 First Row (2 columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Survival Count by Gender")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
    ax1.set_title("Survival Count")
    st.pyplot(fig1)

with col2:
    st.subheader("2️⃣ Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=30, ax=ax2)
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)

# 🔹 Second Row (2 columns)
col3, col4 = st.columns(2)

with col3:
    st.subheader("3️⃣ Survival Rate by Passenger Class")
    fig3, ax3 = plt.subplots()
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", ax=ax3)
    ax3.set_title("Survival Rate by Class")
    st.pyplot(fig3)

with col4:
    st.subheader("4️⃣ Fare vs Age (Colored by Survival)")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=filtered_df, x="Age", y="Fare", hue="Survived", palette="Set1", ax=ax4)
    ax4.set_title("Fare vs Age")
    st.pyplot(fig4)

# 🔹 Third Row (2 columns)
col5, col6 = st.columns(2)

with col5:
    st.subheader("5️⃣ Passenger Count by Embarked Port")
    fig5, ax5 = plt.subplots()
    sns.countplot(data=filtered_df, x="Embarked", hue="Survived", ax=ax5)
    ax5.set_title("Embarked Port vs Survival")
    st.pyplot(fig5)

with col6:
    st.subheader("6️⃣ Correlation Heatmap")
    numeric_cols = ["Age", "Fare", "Pclass", "Survived", "SibSp", "Parch"]
    corr_data = filtered_df[numeric_cols].dropna()
    fig6, ax6 = plt.subplots()
    sns.heatmap(corr_data.corr(), annot=True, cmap="coolwarm", ax=ax6)
    ax6.set_title("Feature Correlation Heatmap")
    st.pyplot(fig6)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
