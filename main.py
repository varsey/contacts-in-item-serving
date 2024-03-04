import dataclasses

import pandas as pd
from fastapi import FastAPI
from lib.model import ModelRunner
from lib.data_loader import DataLoader
from starlette_exporter import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

data_loader = DataLoader()
model_runner = ModelRunner()
print(f'Инициализация готова')

TOTAL_PREDICTIONS = Counter("total", "Total number of predictions")
FOUND_COUNTER = Counter("is_found", "Number of times personal info was detected")


@dataclasses.dataclass
class AdItem:
    title: str
    description: str
    subcategory: str
    category: str


@app.get("/")
async def root():
    return {"message": "This is ML serving app"}


@app.get("/")
def read_healthcheck():
    return {"status": "Green", "version": "0.1.0"}


@app.get("/predict_sample")
def predict():
    test_data = data_loader.load_test_data().sample(1)
    preds = model_runner.get_predicts(test_data)
    print('Using model files: ', model_runner.model_files)
    TOTAL_PREDICTIONS.inc()
    if preds > 0.5:
        FOUND_COUNTER.inc()
    return {
        "text_id": test_data.index.values[0].__str__(),
        "text": test_data.description.values[0].__str__(),
        "has_personal": preds.__str__()
    }


@app.post("/predict")
def predict(item: AdItem):
    test_data = pd.DataFrame([dataclasses.asdict(item)])
    preds = model_runner.get_predicts(test_data)

    return {
        "text_id": test_data.index.values[0].__str__(),
        "text": test_data.description.values[0].__str__(),
        "has_personal": preds.__str__()
    }


@app.get("/retrain/{records}")
def retrain(records: str):
    if int(records) <= 5000:
        train_data = data_loader.load_train_data().sample(int(records))
    else:
        train_data = data_loader.load_train_data()

    test_data = data_loader.load_test_data().sample(100)
    model_runner.retrain(test_data, train_data)

    return {"Status": "Prediction completed"}
