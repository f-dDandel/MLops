import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
import joblib

def train_and_save_model():
    # Загрузка данных
    df = pd.read_csv("house_sales.csv")
    
    # Очистка (ваш код из clear_data)
    df = df.drop(columns=['DocumentDate', 'ym', 'PropertyID', 'ZipCode', 'YrRenovated'])
    df = df.dropna()
    df = df[(df['NbrLivingUnits'] > 0) & (df['NbrLivingUnits'] < 3)]
    # ... остальные фильтры ...

    # Кодирование категориальных признаков
    cat_columns = ['PropertyType', 'NewConstruction']
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    # Масштабирование
    X = df.drop(columns=['AdjSalePrice'])
    y = df['AdjSalePrice']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Обучение модели
    model = Lasso(alpha=0.01, random_state=42)
    model.fit(X_scaled, y)

    # Сохранение модели и scaler
    joblib.dump(model, "lasso_house_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("Model saved as lasso_house_model.joblib")

if __name__ == "__main__":
    train_and_save_model()
