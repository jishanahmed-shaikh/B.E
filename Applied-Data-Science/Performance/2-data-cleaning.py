import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Create sample dataset with missing values
data = {
    'Country': ['USA', None, 'UK', 'India', 'USA', None, 'UK', 'India', 'USA', 'UK'],
    'Age': [25, 30, None, 35, None, 32, 29, 40, 26, None],
    'Salary': [50000, None, 45000, None, 55000, 48000, None, 42000, 52000, 46000],
    'Purchased': ['Yes', 'No', 'Yes', None, 'No', 'Yes', None, 'No', 'Yes', None]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)
print("\nMissing values in each column:")
print(df.isnull().sum())

# 1. Handling Missing Values for Numerical Columns
def handle_numerical_missing_values(df, strategy='mean'):
    numerical_columns = ['Age', 'Salary']
    imputer = SimpleImputer(strategy=strategy)
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    return df

# 2. Handling Missing Values for Categorical Columns
def handle_categorical_missing_values(df, strategy='most_frequent'):
    categorical_columns = ['Country', 'Purchased']
    imputer = SimpleImputer(strategy=strategy)
    df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
    return df

# 3. Label Encoding
def apply_label_encoding(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[f'{col}_encoded'] = le.fit_transform(df[col])
    return df

# 4. One-Hot Encoding
def apply_one_hot_encoding(df, columns):
    return pd.get_dummies(df, columns=columns, prefix=columns)

# Apply all cleaning techniques
# First, handle missing values
df = handle_numerical_missing_values(df)
df = handle_categorical_missing_values(df)

print("\nAfter handling missing values:")
print(df)

# Apply Label Encoding
df = apply_label_encoding(df, ['Country', 'Purchased'])

print("\nAfter Label Encoding:")
print(df)

# Apply One-Hot Encoding
df_onehot = apply_one_hot_encoding(df.drop(['Country_encoded', 'Purchased_encoded'], axis=1), 
                                 ['Country', 'Purchased'])

print("\nAfter One-Hot Encoding:")
print(df_onehot)

# Generate summary statistics for numerical columns
print("\nSummary Statistics after cleaning:")
print(df_onehot.describe())

# Check for any remaining missing values
print("\nRemaining missing values:")
print(df_onehot.isnull().sum())
