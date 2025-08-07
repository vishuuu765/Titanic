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

# ------------------ SIDEBAR FILTERS ------------------

st.sidebar.header("🔍 Filter Options")

# Gender filter
gender = st.sidebar.multiselect("Select Gender", options=df["Sex"].unique(), default=list(df["Sex"].unique()))

# Pclass filter
pclass = st.sidebar.multiselect("Select Passenger Class", options=sorted(df["Pclass"].unique()), default=sorted(df["Pclass"].unique()))

# Age slider filter
age_min = int(df["Age"].min())
age_max = int(df["Age"].max())
age_range = st.sidebar.slider("Select Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

# Fare slider filter
fare_min = int(df["Fare"].min())
fare_max = int(df["Fare"].max())
fare_range = st.sidebar.slider("Select Fare Range", min_value=fare_min, max_value=fare_max, value=(fare_min, fare_max))

# Embarked port filter
embarked = st.sidebar.multiselect("Select Embarked Port", options=df["Embarked"].dropna().unique(), default=list(df["Embarked"].dropna().unique()))

# ------------------ APPLY FILTERS ------------------

filtered_df = df[
    (df["Sex"].isin(gender)) &
    (df["Pclass"].isin(pclass)) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Fare"].between(fare_range[0], fare_range[1])) &
    (df["Embarked"].isin(embarked))
]

# ------------------ DATA PREVIEW ------------------

st.subheader("📋 Filtered Raw Data")
st.dataframe(filtered_df)

# ------------------ VISUALIZATIONS ------------------

# First Row
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

# Second Row
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

# Third Row
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

# ------------------ FOOTER ------------------

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
