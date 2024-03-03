import os

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from starlette_exporter import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter
from sklearn.pipeline import Pipeline
from lib.model import Task1
from lib.run import Test

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

print(f'Инициализация 1')
# MODEL = os.getenv("MODEL", default="baseline.v1")
test = Test(debug=True)
print(f'Инициализация 2')
task1 = Task1()
print(f'Инициализация готова')

SURVIVED_COUNTER = Counter("survived", "Number of survived passengers")
PCLASS_COUNTER = Counter("pclass", "Number of passengers by class", ["pclass"])


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.on_event("startup")
def load_model():
    pass


@app.get("/")
def read_healthcheck():
    return {"status": "Green", "version": "0.2.0"}


@app.get("/predict")
def predict():
    test_data = test.test_data().sample(1)  # val if debug otherwise test
    train_data = test.train_data()

    task1_prediction = pd.DataFrame(columns=['index', 'prediction'])
    task1_prediction['index'] = test_data.index
    task1_prediction['prediction'] = task1.predict(test_data, train_data, force_retrain=False)

    return {
        "text_id": test_data.index.values[0].__str__(),
        "text": test_data.description.values[0].__str__(),
        "has_personal": task1_prediction.prediction.values[0].__str__()
    }

