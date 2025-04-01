import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib

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

def train():
    df = pd.read_csv("./df_clear.csv")

    X, y, scaler = scale_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1],
        'selection': ['cyclic', 'random'],
        'max_iter': [1000, 2000, 5000]
    }
    
    # настройка MLflow
    mlflow.set_experiment("Lasso House Price Prediction")
    
    with mlflow.start_run():
        #обучение
        lasso = Lasso(random_state=42)
        clf = GridSearchCV(lasso, params, cv=5, n_jobs=-1)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        
        # предсказания и метрики
        y_pred = best_model.predict(X_val)
        rmse, mae, r2 = eval_metrics(y_val, y_pred)
        
        # логирование параметров
        mlflow.log_params(best_model.get_params())
        
        # логирование метрик
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        
        # логирование модели
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature
        )
        
        # сохранение модели
        joblib.dump(best_model, "lasso_house_model.pkl")
        mlflow.log_artifact("lasso_house_model.pkl")
        
    return True
