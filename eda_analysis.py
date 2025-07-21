import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_features = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv')
train_labels = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')
train_data = train_features.merge(train_labels, on='id')

# Separate features by type
numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('id')  # Remove ID as it's not a feature
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove('status_group')  # Remove target

print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# Analyze numeric features
print("\n=== NUMERIC FEATURES ANALYSIS ===")
for feature in numeric_features[:10]:  # First 10 numeric features
    print(f"\n{feature}:")
    print(f"  Unique values: {train_data[feature].nunique()}")
    print(f"  Missing: {train_data[feature].isnull().sum()}")
    print(f"  Mean: {train_data[feature].mean():.2f}")
    print(f"  Std: {train_data[feature].std():.2f}")
    print(f"  Min: {train_data[feature].min()}")
    print(f"  Max: {train_data[feature].max()}")

# Analyze date feature
print("\n=== DATE ANALYSIS ===")
train_data['date_recorded'] = pd.to_datetime(train_data['date_recorded'])
train_data['year_recorded'] = train_data['date_recorded'].dt.year
train_data['month_recorded'] = train_data['date_recorded'].dt.month
train_data['day_of_week'] = train_data['date_recorded'].dt.dayofweek

print(f"Date range: {train_data['date_recorded'].min()} to {train_data['date_recorded'].max()}")
print(f"Years range: {train_data['year_recorded'].min()} to {train_data['year_recorded'].max()}")

# Construction year analysis
print("\n=== CONSTRUCTION YEAR ANALYSIS ===")
print(f"Construction year range: {train_data['construction_year'].min()} to {train_data['construction_year'].max()}")
print(f"Zero values in construction_year: {(train_data['construction_year'] == 0).sum()}")

# Geographic features
print("\n=== GEOGRAPHIC FEATURES ===")
print(f"Unique regions: {train_data['region'].nunique()}")
print(f"Unique districts: {train_data['district_code'].nunique()}")
print(f"Unique wards: {train_data['ward'].nunique()}")
print(f"Unique basins: {train_data['basin'].nunique()}")

# Analyze key categorical features
print("\n=== KEY CATEGORICAL FEATURES ===")
important_cats = ['extraction_type', 'management', 'payment', 'water_quality', 'quantity', 'source']
for feature in important_cats:
    print(f"\n{feature} - Top 5 values:")
    print(train_data[feature].value_counts().head())

# Analyze target relationships
print("\n=== TARGET RELATIONSHIPS ===")

# Function to analyze categorical feature vs target
def analyze_cat_vs_target(df, feature, target='status_group'):
    crosstab = pd.crosstab(df[feature], df[target], normalize='index') * 100
    print(f"\n{feature} vs {target} (% distribution):")
    print(crosstab.head(10))
    return crosstab

# Analyze important categorical features vs target
for feature in ['quantity', 'water_quality', 'payment_type', 'extraction_type_class']:
    analyze_cat_vs_target(train_data, feature)

# Analyze numeric features vs target
print("\n=== NUMERIC FEATURES VS TARGET ===")
for feature in ['amount_tsh', 'gps_height', 'population', 'construction_year']:
    if feature in train_data.columns:
        print(f"\n{feature} by status_group:")
        print(train_data.groupby('status_group')[feature].agg(['mean', 'median', 'std']))