"""
FastAPI interface used to make inference on census data using a neural network. The app is deployed on Heroku.

author: Geoffroy de Gournay
date: Mai 15, 2022
"""
import pandas as pd

from starter.ml.model import get_trained_mlp, inference
from starter.ml.data import get_processed_test_data, process_data, get_cat_features
from starter.train_model import model_metrics_slices, model_metrics

from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects
from pydantic import BaseModel, Field
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    # This code is necessary for Heroku to use dvc
    logger.info("Running DVC")
    os.system("dvc config core.no_scm true")
    pull_err = os.system("dvc pull")
    if pull_err != 0:
        exit(f"dvc pull failed, error {pull_err}")
    else:
        logger.info("DVC Pull worked.")
    logger.info('removing dvc files')
    os.system("rm -r .dvc .apt/usr/lib/dvc")


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
    """
    Compute the performance of the model on slices of the data using the categorical feature education.
    """
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
async def api_greeting():
    logger.info("starting GET request")
    return {"greeting": "Welcome! This API predicts income category using Census data."}


@app.post("/predict", response_model=Item)
async def predict(predict_body: CensusItem):
    logger.info("starting POST request")
    model = get_trained_mlp()
    logger.info("get the model")
    data = pd.DataFrame([predict_body.dict(by_alias=True)])
    logger.info('get data from body as a CensusItem object')

    cat_features = get_cat_features(for_api=False)
    x, _, _, _, _ = process_data(data, categorical_features=cat_features,
                                 training=False, encoder=model.encoder, lb=model.lb, scaler=model.scaler)
    logger.info(f'data processed shape: {x.shape}')
    predicted = inference(model, x)

    # Return predicted salary class
    output = Item(predicted_salary_class=predicted[0])

    return output


if __name__ == '__main__':
    model_evaluation()

