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

r = requests.post(
    url='https://geof-census-app.herokuapp.com/predict',
    data=json.dumps(data),
    headers={'content-type': 'application/json'}
)
print(r.json())
