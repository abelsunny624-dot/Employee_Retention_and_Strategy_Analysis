import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

# -----------------------------------------------------
# DATA LOADING
# -----------------------------------------------------

st.sidebar.title("Dataset")

try:
    df = pd.read_csv("cleaned_hr_data.csv")
    st.sidebar.success("Default dataset loaded")
except:
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload dataset to continue")
        st.stop()

# -----------------------------------------------------
# DATA PREPROCESSING
# -----------------------------------------------------

if "Attrition" in df.columns:
    if df["Attrition"].dtype == object:
        df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})

# Age groups
bins = [18,25,35,45,55,60]
labels = ['18-25','26-35','36-45','46-55','56+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# -----------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------

st.sidebar.title("Filters")

department = st.sidebar.multiselect(
    "Department",
    df["Department"].unique(),
    default=df["Department"].unique()
)

overtime = st.sidebar.multiselect(
    "Overtime",
    df["OverTime"].unique(),
    default=df["OverTime"].unique()
)

filtered_df = df[
    (df["Department"].isin(department)) &
    (df["OverTime"].isin(overtime))
]

# -----------------------------------------------------
# PAGE NAVIGATION
# -----------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Summary & KPIs","Visualizations","Prediction"]
)

# =====================================================
# PAGE 1 : SUMMARY + KPI
# =====================================================

if page == "Summary & KPIs":

    st.title("Employee Retention and Strategy Analysis Dashboard")

    total_emp = len(filtered_df)
    left_emp = filtered_df["Attrition"].sum()
    attrition_rate = round(left_emp/total_emp*100,2)

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Employees", total_emp)
    col2.metric("Employees Left", left_emp)
    col3.metric("Attrition Rate %", attrition_rate)

    st.subheader("Summary Statistics")

    st.dataframe(filtered_df.describe())

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head())

# =====================================================
# PAGE 2 : VISUALIZATIONS
# =====================================================
elif page == "Visualizations":

    st.title("Employee Attrition Insights")

    color_map = {0:"#00CC96", 1:"#EF553B"}

    # Q1 DONUT
    st.subheader("Q1 · Attrition Percentage")

    attr = filtered_df["Attrition"].value_counts().reset_index()
    attr.columns = ["Status","Count"]
    attr["Status"] = attr["Status"].map({0:"Stayed",1:"Left"})

    fig = px.pie(attr, names="Status", values="Count", hole=0.5,
                 color="Status",
                 color_discrete_map={"Stayed":"#00CC96","Left":"#EF553B"})
    st.plotly_chart(fig, use_container_width=True)

    rate = round((attr.loc[attr["Status"]=="Left","Count"].values[0]/attr["Count"].sum())*100,2)
    st.success(f"{rate}% of employees have left the company.")

    st.divider()

    # Q2 SORTED BAR
    st.subheader("Q2 · Department Attrition")

    dept = pd.crosstab(filtered_df["Department"], filtered_df["Attrition"], normalize="index")*100
    dept = dept[1].reset_index()
    dept.columns = ["Department","AttritionRate"]
    dept = dept.sort_values("AttritionRate", ascending=False)

    fig = px.bar(dept, x="Department", y="AttritionRate",
                 color="Department")
    fig.update_yaxes(title="Attrition %", ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"{dept.iloc[0]['Department']} has the highest attrition.")

    st.divider()

    # Q3 BOX
    st.subheader("Q3 · Salary vs Attrition")

    fig = px.box(filtered_df,
                 x="Attrition",
                 y="MonthlyIncome",
                 color="Attrition",
                 color_discrete_map=color_map)

    fig.update_xaxes(ticktext=["Stayed","Left"], tickvals=[0,1])
    fig.update_yaxes(title="Monthly Income")

    st.plotly_chart(fig, use_container_width=True)

    sal = filtered_df.groupby("Attrition")["MonthlyIncome"].mean()
    st.success(f"Employees who leave earn less on average.")

    st.divider()

    # Q4 % BAR
    st.subheader("Q4 · Overtime Impact")

    ot = pd.crosstab(filtered_df["OverTime"], filtered_df["Attrition"], normalize="index")*100
    ot = ot[1].reset_index()
    ot.columns = ["OverTime","AttritionRate"]

    fig = px.bar(ot, x="OverTime", y="AttritionRate",
                 color="OverTime",
                 color_discrete_map={"Yes":"#EF553B","No":"#00CC96"})
    fig.update_yaxes(title="Attrition %", ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Employees working overtime leave more (~{round(ot.loc[ot['OverTime']=='Yes','AttritionRate'].values[0],2)}%).")

    st.divider()

    # Q5 STACKED
    st.subheader("Q5 · Age Group")

    age = pd.crosstab(filtered_df["AgeGroup"], filtered_df["Attrition"])

    fig = px.bar(age, barmode="stack",
                 color_discrete_map=color_map)

    st.plotly_chart(fig, use_container_width=True)

    top_age = age[1].idxmax()
    st.success(f"Age group {top_age} has highest attrition.")

    st.divider()

    # Q6 SIMPLE BAR (AVERAGE)
    st.subheader("Q6 · Distance from Home")

    dist = filtered_df.groupby("Attrition")["DistanceFromHome"].mean().reset_index()

    fig = px.bar(dist, x="Attrition", y="DistanceFromHome",
                 color="Attrition",
                 color_discrete_map=color_map)

    fig.update_xaxes(ticktext=["Stayed","Left"], tickvals=[0,1])
    fig.update_yaxes(title="Average Distance")

    st.plotly_chart(fig, use_container_width=True)

    st.success("Employees who live farther tend to leave more.")

    st.divider()

    # Q7 STACKED
    st.subheader("Q7 · Job Satisfaction")

    js = pd.crosstab(filtered_df["JobSatisfaction"], filtered_df["Attrition"])

    fig = px.bar(js, barmode="stack",
                 color_discrete_map=color_map)

    st.plotly_chart(fig, use_container_width=True)

    st.success("Lower satisfaction levels show higher attrition.")

    st.divider()

    # Q8 STACKED
    st.subheader("Q8 · Work-Life Balance")

    wlb = pd.crosstab(filtered_df["WorkLifeBalance"], filtered_df["Attrition"])

    fig = px.bar(wlb, barmode="stack",
                 color_discrete_map=color_map)

    st.plotly_chart(fig, use_container_width=True)

    st.success("Poor work-life balance increases attrition.")

    st.divider()

    # Q9 SIMPLIFIED HEATMAP
    st.subheader("Q9 · Key Relationships")

    cols = ["Age","MonthlyIncome","DistanceFromHome","JobSatisfaction","WorkLifeBalance","Attrition"]
    fig = px.imshow(filtered_df[cols].corr(),
                    color_continuous_scale="Blues")

    st.plotly_chart(fig, use_container_width=True)

    st.success("No single factor dominates — multiple factors influence attrition.")

    st.divider()

    # Q10 BAR AVG
    st.subheader("Q10 · Key Factors Comparison")

    factors = filtered_df.groupby("Attrition")[[
        "MonthlyIncome","JobSatisfaction","WorkLifeBalance","DistanceFromHome","Age"
    ]].mean()

    fig = px.bar(factors.T, barmode="group",
                 color_discrete_map=color_map)

    st.plotly_chart(fig, use_container_width=True)

    st.success("Salary, satisfaction, and distance together influence attrition.")
