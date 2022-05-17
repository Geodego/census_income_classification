import pandas as pd

from starter.ml.model import get_trained_mlp, inference
from starter.ml.data import get_processed_test_data, process_data, get_cat_features
from starter.train_model import model_metrics_slices, model_metrics

from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel, Field


class CensusItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 41,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Divorced",
                "occupation": "Prof-specialty",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2100,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "Cuba"
            }
        }


class Item(BaseModel):
    """Format of API post request response"""
    predicted_salary_class: int

    class Config:
        schema_extra = {
            "example": {
                "predicted_salary_class": 1
            }
        }


def save_education_slices():
    model = get_trained_mlp()
    selected_feature = "education"
    print(model_metrics(model))
    model_metrics_slices(model, selected_feature=selected_feature)


def model_evaluation():
    model = get_trained_mlp()
    x_test, y_test = get_processed_test_data(model.encoder, model.lb, model.scaler)
    precision, recall, f1 = model_metrics(model, x_test, y_test).values()
    print(f'precision: {precision}, recall {recall}, F1: {f1}')


# Instantiate the app.
app = FastAPI()


# Define a GET on the root.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome! This API predicts income category using Census data."}


@app.post("/predict/", response_model=Item)
async def predict(predict_body: CensusItem):
    model = get_trained_mlp()
    data = pd.DataFrame([predict_body.dict()])
    cat_features = get_cat_features(for_api=True)
    x, _, _, _, _ = process_data(data, categorical_features=cat_features,
                                 training=False, encoder=model.encoder, lb=model.lb, scaler=model.scaler)
    predicted = inference(model, x)

    # Return predicted salary class
    output = Item(predicted_salary_class=predicted[0])
    return output


if __name__ == '__main__':
    # save_education_slices()
    # model_evaluation()
    # train_and_save_model(tuning=False, use_saved_model=True)
    a = CensusItem()
    save_education_slices()