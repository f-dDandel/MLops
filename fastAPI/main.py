from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели и scaler
try:
    model = joblib.load("lasso_house_model.joblib")
    scaler = joblib.load("scaler.joblib")
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

app = FastAPI(title="House Price Prediction")

# Модель входных данных (аналогично CarFeatures)
class HouseFeatures(BaseModel):
    PropertyType: str
    NewConstruction: str
    NbrLivingUnits: int
    SqFtLot: float
    SqFtTotLiving: float
    SqFtFinBasement: float
    YrBuilt: int
    LandVal: float
    ImpsVal: float
    Bathrooms: int
    Bedrooms: int
    BldgGrade: int

@app.post("/predict", summary="Predict house price")
async def predict(house: HouseFeatures):
    """
    Предсказывает стоимость дома.
    """
    try:
        # Преобразуем входные данные в DataFrame
        input_data = pd.DataFrame([house.dict()])
        
        # Применяем OrdinalEncoder для категориальных признаков
        cat_columns = ['PropertyType', 'NewConstruction']
        ordinal = OrdinalEncoder()
        input_data[cat_columns] = ordinal.fit_transform(input_data[cat_columns])
        
        # Масштабирование
        scaled_data = scaler.transform(input_data)
        
        # Предсказание
        price = model.predict(scaled_data)[0]
        return {"predicted_price": round(float(price), 2)}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
