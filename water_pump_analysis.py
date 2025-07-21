import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Will use Random Forest only.")
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
train_features = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv')
train_labels = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')
test_features = pd.read_csv('702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv')
submission_format = pd.read_csv('SubmissionFormat.csv')

# Merge training features with labels
train_data = train_features.merge(train_labels, on='id')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_features.shape}")
print(f"\nTarget distribution:")
print(train_data['status_group'].value_counts())
print(f"\nTarget distribution (%):")
print(train_data['status_group'].value_counts(normalize=True) * 100)

# Display basic information
print("\nData types:")
print(train_features.dtypes.value_counts())

print("\nFirst few rows of training data:")
print(train_data.head())

# Check for missing values
print("\nMissing values in training data:")
missing_train = train_features.isnull().sum()
print(missing_train[missing_train > 0])

print("\nMissing values in test data:")
missing_test = test_features.isnull().sum()
print(missing_test[missing_test > 0])