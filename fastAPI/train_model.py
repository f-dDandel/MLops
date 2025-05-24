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
    df = df[(df['AdjSalePrice'] < 1000000) & (df['AdjSalePrice'] > 3000)]
    df = df[(df['SqFtLot'] > 500) & (df['SqFtLot'] < 20000)]
    df = df[(df['SqFtTotLiving'] > 360) & (df['SqFtTotLiving'] < 5100)]
    df = df[(df['SalePrice'] > 3000) & (df['SalePrice'] < 1000000)]
    df = df[df['SqFtFinBasement'] < 1300]
    df = df[(df['YrBuilt'] > 1899) & (df['YrBuilt'] < 2025)]
    df = df[(df['LandVal'] >= 0) & (df['LandVal'] < 800000)]
    df = df[(df['ImpsVal'] >= 0) & (df['ImpsVal'] < 900000)]
    df = df[(df['Bathrooms'] > 0) & (df['Bathrooms'] < 5)]
    df = df[(df['Bedrooms'] > 0) & (df['Bedrooms'] < 7)]
    df = df[(df['BldgGrade'] > 5) & (df['BldgGrade'] < 12)]

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
    print(df.columns)
    joblib.dump(model, "lasso_house_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("Model saved as lasso_house_model.joblib")


if __name__ == "__main__":
    train_and_save_model()
