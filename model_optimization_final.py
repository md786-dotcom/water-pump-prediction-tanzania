import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load all preprocessing functions from water_pump_model.py
import sys
sys.path.append('.')

# Copy preprocessing functions
def engineer_features(df):
    """Engineer features based on EDA insights"""
    df = df.copy()
    
    # 1. Date features
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['year_recorded'] = df['date_recorded'].dt.year
    df['month_recorded'] = df['date_recorded'].dt.month
    df['day_of_year'] = df['date_recorded'].dt.dayofyear
    
    # 2. Age of pump
    df['pump_age'] = df['year_recorded'] - df['construction_year']
    df['pump_age'] = df['pump_age'].apply(lambda x: x if x >= 0 and x < 100 else np.nan)
    
    # 3. GPS features
    df['gps_height_log'] = np.log1p(df['gps_height'].clip(lower=0))
    df['has_gps'] = ((df['longitude'] != 0) & (df['latitude'] != 0)).astype(int)
    
    # 4. Population features
    df['population_log'] = np.log1p(df['population'])
    df['population_per_user'] = df['population'] / (df['num_private'] + 1)
    
    # 5. Categorical feature aggregations
    df['is_dry'] = (df['quantity'] == 'dry').astype(int)
    df['has_enough_water'] = (df['quantity'] == 'enough').astype(int)
    
    # Payment features
    df['never_pays'] = (df['payment'] == 'never pay').astype(int)
    df['pays_annually'] = (df['payment_type'] == 'annually').astype(int)
    
    # Construction year features
    df['construction_year_missing'] = (df['construction_year'] == 0).astype(int)
    df['construction_decade'] = (df['construction_year'] // 10) * 10
    df['construction_decade'] = df['construction_decade'].apply(
        lambda x: x if x >= 1900 and x <= 2020 else 0
    )
    
    # Water quality features
    df['water_quality_unknown'] = (df['water_quality'] == 'unknown').astype(int)
    df['water_quality_good'] = df['water_quality'].isin(['soft', 'fluoride']).astype(int)
    
    # Source features
    df['source_class_ground'] = (df['source_class'] == 'groundwater').astype(int)
    
    # Management features
    df['good_management'] = df['management'].isin(['vwc', 'private operator', 'water board']).astype(int)
    
    # Extraction type features
    df['extraction_type_other'] = (df['extraction_type_class'] == 'other').astype(int)
    df['extraction_type_good'] = df['extraction_type_class'].isin(['gravity', 'handpump', 'rope pump']).astype(int)
    
    return df

def preprocess_data(df, label_encoders=None, fit=True):
    """Preprocess the data"""
    df = df.copy()
    
    # Handle missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('unknown')
    
    # For numeric variables
    df['population'] = df['population'].fillna(df['population'].median())
    df['gps_height'] = df['gps_height'].fillna(df['gps_height'].median())
    df['construction_year'] = df['construction_year'].fillna(0)
    df['amount_tsh'] = df['amount_tsh'].fillna(0)
    
    # Engineer features
    df = engineer_features(df)
    
    # Fill engineered features
    df['pump_age'] = df['pump_age'].fillna(df['pump_age'].median())
    
    # Encode categorical variables
    if label_encoders is None:
        label_encoders = {}
    
    categorical_features = [
        'funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'lga', 'ward',
        'recorded_by', 'scheme_management', 'scheme_name', 'extraction_type', 
        'extraction_type_group', 'extraction_type_class', 'management', 'management_group',
        'payment', 'payment_type', 'water_quality', 'quality_group', 'quantity',
        'quantity_group', 'source', 'source_type', 'source_class', 'waterpoint_type',
        'waterpoint_type_group', 'permit', 'public_meeting'
    ]
    
    for col in categorical_features:
        if col in df.columns:
            if fit:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = df[col].astype(str)
                df[col] = df[col].map(lambda x: x if x in label_encoders[col].classes_ else 'unknown')
                
                if 'unknown' not in label_encoders[col].classes_:
                    classes = list(label_encoders[col].classes_) + ['unknown']
                    label_encoders[col].classes_ = np.array(classes)
                
                df[col] = label_encoders[col].transform(df[col])
    
    if 'date_recorded' in df.columns:
        df = df.drop('date_recorded', axis=1)
    
    return df, label_encoders

# Load data
print("Loading data...")
train_features = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv')
train_labels = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')
test_features = pd.read_csv('702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv')

# Merge and preprocess
train_data = train_features.merge(train_labels, on='id')
X = train_data.drop(['status_group'], axis=1)
y = train_data['status_group']

print("Preprocessing data...")
X_processed, label_encoders = preprocess_data(X)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

feature_cols = [col for col in X_processed.columns if col != 'id']
X_train_full = X_processed[feature_cols]

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# Quick optimization with fewer iterations
print("\nOptimizing Random Forest...")
rf_param_dist = {
    'n_estimators': [200, 300],
    'max_depth': [20, 25, 30],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt'],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_random.fit(X_train, y_train)
print(f"Best parameters: {rf_random.best_params_}")
print(f"Best CV score: {rf_random.best_score_:.4f}")

# Evaluate best model
best_model = rf_random.best_estimator_
val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_accuracy:.4f}")

# Make final predictions
print("\nMaking final predictions...")
test_processed, _ = preprocess_data(test_features, label_encoders, fit=False)
X_test = test_processed[feature_cols]

final_predictions = best_model.predict(X_test)
final_predictions_labels = target_encoder.inverse_transform(final_predictions)

# Create submission
submission = pd.DataFrame({
    'id': test_features['id'],
    'status_group': final_predictions_labels
})

submission.to_csv('water_pump_predictions_final.csv', index=False)
print(f"\nFinal predictions saved to 'water_pump_predictions_final.csv'")
print(f"Submission shape: {submission.shape}")

print("\nPrediction distribution:")
print(submission['status_group'].value_counts())
print("\nSample predictions:")
print(submission.head(10))