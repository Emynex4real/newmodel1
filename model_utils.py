# This file will be used to save and load trained models
import joblib

MODEL_PATHS = {
    'eligibility': 'eligibility_model.pkl',
    'eligibility_scaler': 'eligibility_scaler.pkl',
    'eligibility_features': 'eligibility_features.pkl',
    'scoring': 'scoring_model.pkl',
    'scoring_scaler': 'scoring_scaler.pkl',
    'scoring_features': 'scoring_features.pkl',
}

def save_model(model, scaler, features, prefix):
    joblib.dump(model, MODEL_PATHS[f'{prefix}'])
    joblib.dump(scaler, MODEL_PATHS[f'{prefix}_scaler'])
    joblib.dump(features, MODEL_PATHS[f'{prefix}_features'])

def load_model(prefix):
    try:
        model = joblib.load(MODEL_PATHS[f'{prefix}'])
        scaler = joblib.load(MODEL_PATHS[f'{prefix}_scaler'])
        features = joblib.load(MODEL_PATHS[f'{prefix}_features'])
        return model, scaler, features
    except Exception:
        return None, None, None
