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
    x, y = df.drop(columns=['AdjSalePrice']), df['AdjSalePrice']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    # загрузка данных
    df = pd.read_csv("./df_clear.csv")
    
    # масштабирование данных
    X, Y = scale_frame(df)
    
    # разделение данных
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    # параметры для GridSearchCV
    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1]}
    
    # настройка MLflow
    mlflow.set_experiment("Lasso Model House Sales")
    
    with mlflow.start_run():
        # обучение модели
        lasso = Lasso(random_state=42)
        clf = GridSearchCV(lasso, params, cv=5)
        clf.fit(X_train, y_train)
        
        # лучшая модель
        best = clf.best_estimator_
        
        # предсказания
        y_pred = best.predict(X_val)
        
        # оценка метрик
        (rmse, mae, r2) = eval_metrics(y_val, y_pred)
        
        # логирование параметров и метрик
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # логирование модели
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        # сохранение модели
        joblib.dump(best, "lasso_model.pkl")
    
    # получение пути к лучшей модели
    df_runs = mlflow.search_runs()
    path2model = df_runs.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + '/model'
    print(path2model)