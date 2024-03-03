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


@app.post("/predict")
def predict():
    test_data = pd.DataFrame([{
        'title': 'Мотоблок aurora country 6500 multi-shift',
        'description': ' НОВЫЙ!!!/Объем двигателя389 см3Мощность двигателя13 л.с.Тип топливаБензиновыйСцеплениеДисковоеКоличество скоростей5-вперед,4-назадРеверс (задний ход)ЕстьШирина обработки170 смГлубина обработки30 смЕмкость топливного бака6.2 лОбъем масляного картера1.1 лСистема пускаручной стартерПодключение навесного оборудованияВОМКолёса в комплекте2.00-14Вес164 кгГабариты1800х1100х800 мм Образец.ПроизводительAuroraМодельCOUNTRY 1500 MULTI-SHIFT ',
        'subcategory': 'Ремонт и строительство',
        'category': 'Для дома и дачи',
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
