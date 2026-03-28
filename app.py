from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

def load_model():
    try:
        file = open("Price_model.pkl", "rb")
        model = joblib.load(file)
        return model
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail="load model error")
    
def make_pred(data):
    try:
        model = load_model()
        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        return float(pred)
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail="prediction error")
    
app = FastAPI()

class autodata(BaseModel):
    symboling: int
    normalized_losses :float
    make :str
    fuel_type :str
    aspiration :str
    num_of_doors :str
    body_style :str
    drive_wheels :str
    engine_location :str
    wheel_base :float
    length :float
    width :float
    height :float
    curb_weight :int
    engine_type :str
    num_of_cylinders :str
    engine_size :int
    fuel_system :str
    bore :float
    stroke :float
    compression_ratio :float
    horsepower :float
    peak_rpm :float
    city_mpg :int
    highway_mpg :int

@app.post('/predict')
def predict(data:autodata):
    try:
        result = make_pred(data.dict())
        return{
            "Prediction_Price": float(result)
        }
    except Exception as e:
        print(str(e))
        raise