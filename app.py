import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Set background image from local file (ship.Titanic.jpg)
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function
set_background("ship.Titanic.jpg")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
gender = st.sidebar.selectbox("Select Gender", options=["All"] + list(df["Sex"].dropna().unique()))
pclass = st.sidebar.selectbox("Select Passenger Class", options=["All"] + sorted(df["Pclass"].dropna().unique()))
embarked = st.sidebar.multiselect("Select Embarked Port", options=df["Embarked"].dropna().unique(), default=df["Embarked"].dropna().unique())
age_range = st.sidebar.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (10, 50))

# Apply filters
filtered_df = df.copy()
if gender != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == gender]
if pclass != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass]
filtered_df = filtered_df[
    (filtered_df["Embarked"].isin(embarked)) &
    (filtered_df["Age"].between(age_range[0], age_range[1]))
]

# Show Data
if st.checkbox("Show Filtered Raw Data"):
    st.dataframe(filtered_df)

# 2x3 Layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("Survival Count by Gender")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x="Sex", hue="Sex", ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=30, ax=ax2)
    st.pyplot(fig2)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Survival Rate by Passenger Class")
    fig3, ax3 = plt.subplots()
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", ax=ax3)
    st.pyplot(fig3)

with col4:
    st.subheader("Fare vs Age Scatter Plot")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=filtered_df, x="Age", y="Fare", hue="Sex", ax=ax4)
    st.pyplot(fig4)

col5, col6 = st.columns(2)
with col5:
    st.subheader("Passenger Count by Embarked Port")
    fig5, ax5 = plt.subplots()
    sns.countplot(data=filtered_df, x="Embarked", hue="Embarked", ax=ax5)
    st.pyplot(fig5)

with col6:
    st.subheader("Survival by Pclass and Gender")
    fig6, ax6 = plt.subplots()
    sns.countplot(data=filtered_df, x="Pclass", hue="Sex", ax=ax6)
    st.pyplot(fig6)
