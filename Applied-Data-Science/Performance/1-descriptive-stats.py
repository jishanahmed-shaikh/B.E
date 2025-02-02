import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample BMI dataset
data = {
    'Height': [170, 175, 160, 180, 165, 172, 168, 185, 169, 171],
    'Weight': [68, 75, 55, 85, 60, 70, 65, 90, 63, 69],
    'Age': [25, 30, 28, 35, 27, 32, 29, 40, 26, 31],
    'BMI': [23.5, 24.5, 21.5, 26.2, 22.0, 23.7, 23.0, 26.3, 22.1, 23.6],
    'Gender': ['M', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M']
}

df = pd.DataFrame(data)

# 1. Measures of Central Tendency
def calculate_central_tendency(df, numeric_columns):
    print("MEASURES OF CENTRAL TENDENCY")
    print("-" * 50)
    for col in numeric_columns:
        print(f"\nStatistics for {col}:")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Median: {df[col].median():.2f}")
        print(f"Mode: {df[col].mode()[0]:.2f}")

# 2. Measures of Spread
def calculate_spread(df, numeric_columns):
    print("\nMEASURES OF SPREAD")
    print("-" * 50)
    for col in numeric_columns:
        print(f"\nStatistics for {col}:")
        print(f"Standard Deviation: {df[col].std():.2f}")
        print(f"Variance: {df[col].var():.2f}")
        print(f"Range: {df[col].max() - df[col].min():.2f}")
        print(f"IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")

# 3. Measures of Shape
def calculate_shape(df, numeric_columns):
    print("\nMEASURES OF SHAPE")
    print("-" * 50)
    for col in numeric_columns:
        print(f"\nStatistics for {col}:")
        print(f"Skewness: {df[col].skew():.2f}")
        print(f"Kurtosis: {df[col].kurtosis():.2f}")

# 4. Create visualizations
def create_visualizations(df, numeric_columns):
    # Create boxplots
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(1, len(numeric_columns), i)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

    # Create histograms
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(1, len(numeric_columns), i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

# Main analysis
numeric_columns = ['Height', 'Weight', 'Age', 'BMI']

# Run all analyses
calculate_central_tendency(df, numeric_columns)
calculate_spread(df, numeric_columns)
calculate_shape(df, numeric_columns)
create_visualizations(df, numeric_columns)

# Generate summary statistics
print("\nCOMPLETE SUMMARY STATISTICS")
print("-" * 50)
print(df.describe())
