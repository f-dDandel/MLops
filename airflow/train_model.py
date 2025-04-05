import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib
from pathlib import Path
import os

def scale_frame(frame):
    df = frame.copy()
    X = df.drop(columns=['AdjSalePrice'])
    y = df['AdjSalePrice']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def save_model(model, model_dir="models"):
    """Безопасное сохранение модели"""
    model_dir = Path(file).parent / model_dir
    model_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = model_dir / "lasso_house_model.pkl"
    joblib.dump(model, model_path)
    return model_path

def train(**context):
    """Основная функция обучения"""
    ti = context['ti']
    json_data = ti.xcom_pull(task_ids='clean_house_data')
    df = pd.read_json(json_data, orient='split')

    X, y, scaler = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Lasso House Price Prediction")
    
    with mlflow.start_run():
        # обучение модели
        params = {
            'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1],
            'selection': ['cyclic', 'random'],
            'max_iter': [1000, 2000, 5000]
        }
        
        lasso = Lasso(random_state=42)
        clf = GridSearchCV(lasso, params, cv=5, n_jobs=-1)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        
        # логирование
        y_pred = best_model.predict(X_val)
        rmse, mae, r2 = eval_metrics(y_val, y_pred)
        
        mlflow.log_params(best_model.get_params())
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        
        # сохранение модели
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        #сохранение в файл
        model_path = save_model(best_model)
        mlflow.log_artifact(str(model_path))
        
    return f"Model trained and saved to {model_path}"
