import pandas as pd
from fastapi import FastAPI
from starlette_exporter import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter
from lib.model import ModelRunner
from lib.run import DataLoader

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

data_loader = DataLoader(debug=True)
model_runner = ModelRunner()
print(f'Инициализация готова')

SURVIVED_COUNTER = Counter("survived", "Number of survived passengers")
PCLASS_COUNTER = Counter("pclass", "Number of passengers by class", ["pclass"])


@app.get("/")
async def root():
    return {"message": "This is ML serving app"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.on_event("startup")
def load_model():
    pass


@app.get("/")
def read_healthcheck():
    return {"status": "Green", "version": "0.1.0"}


@app.get("/predict")
def predict():
    test_data = data_loader.load_test_data().sample(1)  # val if debug otherwise test
    train_data = data_loader.load_train_data()

    task1_prediction = pd.DataFrame(columns=['index', 'prediction'])
    task1_prediction['index'] = test_data.index
    task1_prediction['prediction'] = model_runner.predict(test_data, train_data, force_retrain=False)

    return {
        "text_id": test_data.index.values[0].__str__(),
        "text": test_data.description.values[0].__str__(),
        "has_personal": task1_prediction.prediction.values[0].__str__()
    }

