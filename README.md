#  Employee Retention & Strategy Analysis Dashboard

A complete end-to-end Data Analytics project focused on understanding employee attrition and identifying key factors that influence retention.  
This project uses **Python, Streamlit, and Machine Learning** to deliver an interactive and user-friendly dashboard.

---

##  Project Objective

The goal of this project is to:

- Analyze employee attrition patterns
- Identify key factors affecting employee turnover
- Provide actionable insights for HR decision-making
- Predict whether an employee is likely to leave

---

##  Dataset

- Source: IBM HR Analytics Dataset (modified & cleaned)
- File used: `cleaned_hr_data.csv`
- Key features:
  - Age
  - Department
  - Monthly Income
  - Job Satisfaction
  - Work-Life Balance
  - Distance From Home
  - OverTime
  - Attrition (Target Variable)

---

##  Tech Stack

- **Python**
- **Pandas & NumPy** → Data processing
- **Plotly** → Interactive visualizations
- **Streamlit** → Dashboard UI
- **Scikit-learn** → Machine Learning model

---

##  Dashboard Features

###  1. Summary & KPIs
- Total Employees
- Employees Left
- Attrition Rate (%)
- Dataset overview

---

###  2. Visual Analysis

The dashboard answers key business questions:

####  Attrition Overview
- What percentage of employees leave?

####  Department Analysis
- Which department has the highest attrition?

####  Salary Impact
- Do lower-paid employees leave more?

####  Overtime Impact
- Does overtime increase attrition risk?

####  Demographics
- Which age group leaves the most?

####  Behavioral Factors
- Job Satisfaction vs Attrition
- Work-Life Balance vs Attrition

####  Lifestyle Factors
- Distance from home and its impact on attrition

####  Correlation Analysis
- Relationship between multiple variables

---

###  3. Prediction Module

- Uses **Random Forest Classifier**
- Predicts whether an employee is likely to leave
- Inputs:
  - Age
  - Salary
  - Distance
  - Job Satisfaction
  - Work-Life Balance

---

##  Key Insights

- Employees with **lower salaries** are more likely to leave
- **Overtime** significantly increases attrition risk
- Employees aged **26–35** show higher attrition
- **Low job satisfaction** strongly correlates with attrition
- **Poor work-life balance** contributes to employee exits
- Employees living **farther from the workplace** tend to leave more

---
