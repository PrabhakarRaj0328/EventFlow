import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
import json

def train_xgboost(input_file="data/processed_dataset.csv", model_output="models/xgb_model.json", metrics_output="results/cv_metrics.json"):
    print("Loading processed dataset...")
    df = pd.read_csv(input_file)
    
    # Ensure it's sorted by time for TimeSeriesSplit
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df = df.sort_values(by=['start_datetime', 'zone', 'junction'])
    
    # Define features and target
    features = ['zone_encoded', 'junction_encoded', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    target = 'smoothed_count'
    
    # Time-based split for final evaluation (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Save test set for evaluate.py
    test_df.to_csv("data/test_dataset.csv", index=False)
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    print(f"Features: {features}")
    print(f"Target: {target}")
    
    # Time-series cross-validation on train set
    tscv = TimeSeriesSplit(n_splits=3)
    
    # XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Hyperparameter grid focused on regularization
    param_grid = {
        'max_depth': [3, 5, 7],
        'reg_alpha': [0.1, 1.0, 10.0],   # L1 regularization (alpha)
        'reg_lambda': [0.1, 1.0, 10.0],  # L2 regularization (lambda)
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200]
    }
    
    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        cv=tscv,
        scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error'],
        refit='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Save best CV metrics
    best_index = grid_search.best_index_
    cv_rmse = -grid_search.cv_results_['mean_test_neg_root_mean_squared_error'][best_index]
    cv_mae = -grid_search.cv_results_['mean_test_neg_mean_absolute_error'][best_index]
    
    print(f"CV RMSE: {cv_rmse:.4f}")
    print(f"CV MAE: {cv_mae:.4f}")
    
    metrics = {
        'best_params': grid_search.best_params_,
        'cv_rmse': cv_rmse,
        'cv_mae': cv_mae
    }
    
    with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saving model to {model_output}...")
    best_model.save_model(model_output)
    
    # Save the feature names for evaluation
    joblib.dump(features, "models/features.joblib")
    print("Training complete!")

if __name__ == "__main__":
    train_xgboost()
