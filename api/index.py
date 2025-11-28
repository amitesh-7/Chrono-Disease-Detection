"""
Disease Classification API for Vercel Serverless
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Model paths (Vercel stores files in /var/task/)
MODEL_DIR = 'models'

# Global model cache
_models = {}

def load_models():
    """Load models on cold start"""
    global _models
    
    if not _models:
        try:
            xgb_path = os.path.join(MODEL_DIR, 'xgb_model.joblib')
            encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
            
            _models['xgb'] = joblib.load(xgb_path)
            _models['encoder'] = joblib.load(encoder_path)
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            _models = None
    
    return _models

# Feature names
FEATURE_NAMES = [
    'age', 'gender', 'smoker', 'heart_rate', 'blood_pressure', 'cholesterol_level',
    'fever', 'cough', 'fatigue', 'shortness_of_breath', 'headache', 'runny_nose',
    'sore_throat', 'chest_pain', 'body_ache', 'nausea', 'vomiting', 'diarrhea',
    'dizziness', 'chills', 'loss_of_smell', 'loss_of_taste', 'wheezing', 'rash',
    'eye_irritation', 'ear_pain', 'sweating', 'joint_pain', 'abdominal_pain',
    'back_pain', 'blurred_vision', 'dry_cough', 'wet_cough', 'sinus_pressure',
    'sneezing', 'rapid_heartbeat', 'slow_heartbeat', 'dehydration',
    'loss_of_appetite', 'sleep_disturbance', 'anxiety', 'irritability',
    'muscle_spasm', 'skin_redness', 'itchiness', 'breathing_difficulty'
]

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for disease prediction"""
    models = load_models()
    
    if not models:
        return jsonify({
            'error': 'Models not available. Please ensure models are deployed.'
        }), 500
    
    try:
        data = request.get_json()
        model_choice = data.get('model', 'xgboost').lower()
        
        # Build feature array
        features = []
        for feature in FEATURE_NAMES:
            value = data.get(feature)
            if value is None:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            features.append(value)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        # Use XGBoost model only
        model = models['xgb']
        model_choice = 'xgboost'
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get disease name and confidence
        disease = models['encoder'].inverse_transform([prediction])[0]
        confidence = float(prediction_proba[prediction]) * 100
        
        # Get all probabilities
        all_diseases = models['encoder'].classes_
        probabilities = {
            disease: float(prob) * 100 
            for disease, prob in zip(all_diseases, prediction_proba)
        }
        
        return jsonify({
            'success': True,
            'prediction': disease,
            'confidence': round(confidence, 2),
            'model_used': model_choice,
            'all_probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    models = load_models()
    models_loaded = models is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'models_not_loaded',
        'models_available': models_loaded
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Return list of required features"""
    return jsonify({
        'features': FEATURE_NAMES,
        'total_features': len(FEATURE_NAMES)
    })

# Vercel handler
def handler(request):
    """Vercel serverless handler"""
    with app.request_context(request.environ):
        return app.full_dispatch_request()
