from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import Ridge
import pickle, logging, io
from contextlib import asynccontextmanager
import numpy as np
from sklearn.compose import ColumnTransformer
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import StringIO

logging.basicConfig(level=logging.INFO)

model: Ridge = None
preprocessor: ColumnTransformer = None

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Start context loading")
    global model, preprocessor
    try:
        with open('ridge.pkl', 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Модель загружена из ridge.pkl.")

        with open('column_transformer.pkl', 'rb') as file:
            preprocessor = pickle.load(file)
        logging.info(f"ColumnTransformer загружен из column_transformer.pkl.")
        
        yield
    except Exception as e:
        logging.error(f"Ошибка при загрузке: {e}")
    finally:
        model = None
        preprocessor = None

app = FastAPI(lifespan=lifespan)

@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    return get_predict_item(item)


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)) -> StreamingResponse:
    contents = file.file.read()
    s = str(contents,'utf-8')
    data = StringIO(s) 
    df = pd.read_csv(data)
    data.close()
    file.file.close()
    predictions = get_predict_items(df)
    df['predictions'] = predictions
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return StreamingResponse(io.StringIO(csv_buffer.getvalue()), 
                             media_type="text/csv", 
                             headers={"Content-Disposition": "attachment; filename=predictions.csv"})

def get_predict_item(item) -> float:
    features = create_df_from_item(item)
    prediction = model.predict(features)
    return float(prediction[0])

def get_predict_items(items) -> List[float]:
    features = create_df_from_items(items)
    predictions = model.predict(features)
    return predictions.tolist()

def create_df_from_items(items) -> pd.DataFrame:
    items.drop("selling_price", axis=1, inplace=True)
    df = preprocess_items(items)
    df = transform_item(df)
    return df

def create_df_from_item(item: Item) -> pd.DataFrame:
    item_dict = item.model_dump()
    df = pd.DataFrame([item_dict])
    df = preprocess_items(df)
    df = transform_item(df)
    return df

def transform_item(features: pd.DataFrame) -> pd.DataFrame:
    return preprocessor.transform(features)

def parse_torque(value):
    if pd.notna(value):
        kgm_match = pd.Series(value).str.extract(r'(\d+(\.\d+)?)\s*kgm')
        nm_match = pd.Series(value).str.extract(r'(\d+(\.\d+)?)\s*nm') #
        
        if not kgm_match.empty and kgm_match[0].notna().any():
            kgm_value = float(kgm_match[0].iloc[0])
            return kgm_value * 9.80665
        elif not nm_match.empty and nm_match[0].notna().any():
            return float(nm_match[0].iloc[0])
    
    return None

def preprocess_items(items: pd.DataFrame) -> pd.DataFrame:
    items['mileage'] = items['mileage'].str.extract(r'(\d+(\.\d+)?)')[0].astype(float)
    items['engine'] = items['engine'].str.replace(' CC', '', regex=False).astype(float)
    items['max_power'] = pd.to_numeric(items['max_power'].str.replace(' bhp', '', regex=False), errors='coerce')
    items['torque'] = items['torque'].apply(parse_torque).astype(float)
    items.rename(columns={'mileage': 'mileage (kmpl)'}, inplace=True)
    items.rename(columns={'engine': 'engine (СС)'}, inplace=True)
    items.rename(columns={'max_power': 'max_power (bhp)'}, inplace=True)
    items.rename(columns={'torque': 'torque (Nm)'}, inplace=True)
    items['engine (СС)'] = items['engine (СС)'].astype(int)
    items['seats'] = items['seats'].astype(int)
    return items
