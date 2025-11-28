"""
Disease Classification API for Render
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder='public')

# Model paths
MODEL_DIR = 'models'

# Global model cache
_models = {}

def load_models():
    """Load models on startup"""
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

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('public', 'index.html')

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
        
        # Build feature array
        features = []
        for feature in FEATURE_NAMES:
            value = data.get(feature)
            if value is None:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            features.append(value)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        # Use XGBoost model
        model = models['xgb']
        
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
            'model_used': 'xgboost',
            'all_probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    models = load_models()
    return jsonify({
        'status': 'healthy',
        'models_available': models is not None
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get list of required features"""
    return jsonify({
        'features': FEATURE_NAMES,
        'count': len(FEATURE_NAMES)
    })

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    try:
        return send_from_directory('public', path)
    except:
        return send_from_directory('public', 'index.html')

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Not Found', 'message': 'The requested resource was not found'}), 404

if __name__ == '__main__':
    load_models()  # Pre-load models
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
