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
