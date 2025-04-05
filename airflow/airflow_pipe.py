import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from train_model import train
from pathlib import Path
from airflow.operators.empty import EmptyOperator


def download_data(**context):
    dag_dir = Path(file).parent
    csv_path = dag_dir / 'house_sales.csv'
    df = pd.read_csv(csv_path, delimiter=',')

    print("Initial data shape:", df.shape)
    return df.to_json(orient='split')

def clear_data(**context):
    ti = context['ti']
    json_data = ti.xcom_pull(task_ids='download_house_data')
    df = pd.read_json(json_data, orient='split')

    df = df.drop(columns=['DocumentDate', 'ym', 'PropertyID', 'ZipCode', 'YrRenovated'])
    df = df.dropna()

    cat_columns = ['PropertyType', 'NewConstruction']
    num_columns = ['NbrLivingUnits', 'AdjSalePrice', 'SqFtLot', 'SqFtTotLiving', 
                  'SalePrice', 'SqFtFinBasement', 'YrBuilt', 'LandVal', 'ImpsVal',
                  'Bathrooms', 'Bedrooms', 'BldgGrade']

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

    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    

    print("Cleaned data shape:", df.shape)
    return df.to_json(orient='split')

    
# определение DAG
dag_houses = DAG(
    dag_id="train_house_pipeline",
    start_date=datetime(2025, 3, 1),
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
)

# создание задач
download_task = PythonOperator(
    task_id="download_house_data",
    python_callable=download_data,
    dag=dag_houses,
)

clear_task = PythonOperator(
    task_id="clean_house_data",
    python_callable=clear_data,
    dag=dag_houses,
)

train_task = PythonOperator(
    task_id="train_house_model",
    python_callable=train,
    dag=dag_houses,
)

# определение порядка выполнения
download_task >> clear_task >> train_task
