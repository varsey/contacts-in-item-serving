import pandas as pd
from fastapi import FastAPI
from lib.model import ModelRunner
from lib.data_loader import DataLoader
from starlette_exporter import PrometheusMiddleware, handle_metrics

app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

data_loader = DataLoader()
model_runner = ModelRunner()
print(f'Инициализация готова')


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


@app.get("/predict_sample")
def predict():
    test_data = data_loader.load_test_data().sample(1)
    preds = model_runner.get_predicts(test_data)

    return {
        "text_id": test_data.index.values[0].__str__(),
        "text": test_data.description.values[0].__str__(),
        "has_personal": preds.__str__()
    }


@app.get("/predict")
def predict():
    test_data = pd.DataFrame([{
        'title': 'Honda VFR 800 2004 г.в',
        'description': 'Honda VFR 800 2004 г.в	Мот в отличном состоянии для своих лет, Родной пластик, новая резина перед-зад, родной пробег 37 тыс, привезен из Германии в 3043 году, на тер. РФ я второй собственник, торг минимальный',
        'subcategory': 'Мотоциклы и мототехника',
        'category': 'Транспорт',
    }])
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

    test_data = data_loader.load_test_data().sample(1)

    model_runner.retrain(test_data, train_data)

    return {"Status": "Prediction completed"}
