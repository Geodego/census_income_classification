import requests
import json

data = {
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
response1 = requests.get("https://geof-census-app.herokuapp.com/")
print(f"status get: {response1.status_code}")
from starter.ml.data import get_clean_data
df = get_clean_data()
data = df.iloc[0].to_dict()
r = requests.post(
    url='https://geof-census-app.herokuapp.com/predict',
    json=data,
)
print(r.json())
