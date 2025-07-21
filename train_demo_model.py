"""
Train a lightweight model for the Streamlit demo
This creates a smaller model file suitable for deployment
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import joblib

# For demo purposes, create a simple rule-based classifier
class SimpleWaterPumpClassifier:
    """Simplified classifier for demo deployment"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['functional', 'functional needs repair', 'non functional'])
        
    def predict(self, features):
        """
        Predict based on key rules discovered in EDA
        features: dict with keys like 'quantity', 'water_quality', etc.
        """
        # Rule 1: Dry pumps are almost always non-functional
        if features.get('quantity') == 'dry':
            return 'non functional', 0.97
            
        # Rule 2: Unknown water quality strongly indicates non-functional
        if features.get('water_quality') == 'unknown':
            return 'non functional', 0.84
            
        # Rule 3: Enough water + regular payment = likely functional
        if (features.get('quantity') == 'enough' and 
            features.get('payment_type') in ['annually', 'monthly']):
            return 'functional', 0.75
            
        # Rule 4: Other extraction type = likely non-functional
        if features.get('extraction_type') == 'other':
            return 'non functional', 0.80
            
        # Rule 5: Enough water generally means functional
        if features.get('quantity') == 'enough':
            return 'functional', 0.65
            
        # Default: Use base rates
        prob = np.random.random()
        if prob < 0.543:  # 54.3% functional
            return 'functional', 0.60
        elif prob < 0.927:  # 38.4% non-functional
            return 'non functional', 0.60
        else:  # 7.3% needs repair
            return 'functional needs repair', 0.60
    
    def predict_proba(self, features):
        """Return probability distribution"""
        prediction, confidence = self.predict(features)
        
        # Create probability array
        proba = np.array([0.0, 0.0, 0.0])
        
        if prediction == 'functional':
            proba[0] = confidence
            proba[2] = (1 - confidence) * 0.7  # non-functional
            proba[1] = (1 - confidence) * 0.3  # needs repair
        elif prediction == 'non functional':
            proba[2] = confidence
            proba[0] = (1 - confidence) * 0.6  # functional
            proba[1] = (1 - confidence) * 0.4  # needs repair
        else:  # needs repair
            proba[1] = confidence
            proba[0] = (1 - confidence) * 0.5
            proba[2] = (1 - confidence) * 0.5
            
        return proba

# Create and save the model
model = SimpleWaterPumpClassifier()

# Save model
with open('demo_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Demo model saved successfully!")

# Also save feature list for reference
features_info = {
    'numeric_features': ['longitude', 'latitude', 'gps_height', 'population', 'construction_year'],
    'categorical_features': {
        'quantity': ['enough', 'insufficient', 'dry', 'seasonal', 'unknown'],
        'water_quality': ['soft', 'salty', 'milky', 'coloured', 'fluoride', 'unknown'],
        'payment_type': ['never pay', 'per bucket', 'monthly', 'annually', 'on failure', 'unknown'],
        'extraction_type': ['gravity', 'handpump', 'motorpump', 'rope pump', 'submersible', 'other'],
        'management': ['vwc', 'wug', 'water board', 'private operator', 'company', 'other'],
        'source': ['spring', 'river', 'shallow well', 'borehole', 'rainwater harvesting', 'other']
    },
    'key_insights': {
        'dry_pumps': '97% are non-functional',
        'unknown_quality': '84% are non-functional',
        'enough_water': '65% are functional',
        'annual_payment': '75% are functional'
    }
}

with open('features_info.pkl', 'wb') as f:
    pickle.dump(features_info, f)

print("Feature information saved successfully!")