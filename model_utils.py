import joblib
import os

MODEL_PATHS = {
    'eligibility': '/tmp/eligibility_model.pkl',
    'eligibility_scaler': '/tmp/eligibility_scaler.pkl',
    'eligibility_features': '/tmp/eligibility_features.pkl',
    'scoring': '/tmp/scoring_model.pkl',
    'scoring_scaler': '/tmp/scoring_scaler.pkl',
    'scoring_features': '/tmp/scoring_features.pkl',
}

def save_model(model, scaler, features, prefix):
    os.makedirs('/tmp', exist_ok=True)
    joblib.dump(model, MODEL_PATHS[f'{prefix}'])
    joblib.dump(scaler, MODEL_PATHS[f'{prefix}_scaler'])
    joblib.dump(features, MODEL_PATHS[f'{prefix}_features'])

def load_model(prefix):
    try:
        model = joblib.load(MODEL_PATHS[f'{prefix}'])
        scaler = joblib.load(MODEL_PATHS[f'{prefix}_scaler'])
        features = joblib.load(MODEL_PATHS[f'{prefix}_features'])
        return model, scaler, features
    except Exception as e:
        print(f"Error loading model {prefix}: {str(e)}")
        return None, None, None