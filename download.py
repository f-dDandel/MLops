import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    # загрузка данных
    df = pd.read_csv('house_sales.csv', delimiter=',')
    df.to_csv("house_sales_processed.csv", index=False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    # удаление ненужных столбцов
    df = df.drop(columns=['DocumentDate', 'ym', 'PropertyID', 'ZipCode', 'YrRenovated'])
    df = df.dropna()
    
    # категориальные и числовые столбцы
    cat_columns = ['PropertyType', 'NewConstruction']
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].astype(np.float64)
    
    # очистка данных от слишком больших и нереалистичных значений
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
    
    # сброс индексов
    df = df.reset_index(drop=True)
    
    # кодирование категориальных признаков
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    # сохранение очищенных данных
    df.to_csv('df_clear.csv', index=False)
    return True

# вызов функций
download_data()
clear_data("house_sales_processed.csv")
