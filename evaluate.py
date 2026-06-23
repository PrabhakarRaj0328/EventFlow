import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import json

def evaluate_model(test_file="data/test_dataset.csv", model_file="models/xgb_model.json", features_file="models/features.joblib", metrics_output="results/test_metrics.json"):
    print("Loading test dataset...")
    test_df = pd.read_csv(test_file)
    
    print("Loading model and features...")
    features = joblib.load(features_file)
    
    model = xgb.XGBRegressor()
    model.load_model(model_file)
    
    X_test = test_df[features]
    y_test = test_df['smoothed_count']
    
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # Ensure no negative predictions if counts can't be negative
    predictions = np.clip(predictions, a_min=0, a_max=None)
    
    print("Calculating metrics...")
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    metrics = {
        'test_rmse': rmse,
        'test_mae': mae
    }
    
    with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Test metrics saved to {metrics_output}")

if __name__ == "__main__":
    evaluate_model()
