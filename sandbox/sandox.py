import pandas as pd
import numpy as np

# ====================================
# Create Dataframe
data = {
    "A": [],
    "B": []
}

df = pd.DataFrame(data)

print(df)
# ====================================
# Read csv and print first 5
titanic_data = pd.read_csv("./titanic_dataset.csv")

titanic_data.head()
# ====================================
# Read dataset column names, dataset.
titanic_data.info()

# ====================================
# Display the dataset's missing values
print(titanic_data.isnull().sum())

# ====================================
# Display specific columns
print(titanic_data.loc[0:9, ['Name','Age','Fare']])

# ====================================
# Displays descriptive statistics of the dataset
print(titanic_data.describe())

# ====================================
# Remove rows with specific missing values
print(titanic_data.dropna(subset=['Age']))

# ====================================
# Remove duplicates
titanic_data.drop_duplicates()

# ====================================
# Compute and display the correlation matrix of the dataset
try:
    correlation_matrix = titanic_data.corr()
    print(correlation_matrix)
except ValueError:
    print(" Dataframe should not contain a row or a cell that is a string")
# ValueError: could not convert string to float: 'Snyder, Mrs. John Pillsbury (Nelle Stevenson)'
# InSummary: Dataframe should not contain a row or a cell that is a string

# Other solution incase: 
# titanic_data.select_dtypes(include=[float,int])

# ====================================

# **Case Study 1: Iris Flower Classification** 🌸  

# ### **Background**  
# A botanical research institute wants to develop an automated system that classifies different species of **iris flowers** based on their **sepal and petal measurements**.  The dataset consists of **150 samples**, labeled as **Setosa, Versicolor, or Virginica**.  

# ### **Problem Statement**  
# Can we use **sepal and petal dimensions** to correctly classify the **species of an iris flower**?  


# ### **Task Description**  

# #### **1. Data Exploration**  
# - Load the dataset and display the first few rows.  
# - Identify any missing or inconsistent values.  

# #### **2. Data Cleaning**  
# - Check for missing values and handle them appropriately.  
# - Convert categorical species labels into a format suitable for analysis.  

# #### **3. Basic Data Analysis**  
# - Find the average sepal and petal dimensions for each species.  
# - Identify correlations between different flower measurements.  

# #### **4. Visualization**  
# - Create simple visualizations (e.g., histograms, scatter plots) to understand data distribution.  

# #### **5. Insights & Interpretation**  
# - Summarize key findings, such as which features best distinguish flower species.  