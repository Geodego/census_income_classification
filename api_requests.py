import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

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
url = 'https://geof-census-app.herokuapp.com/'
# url = "http://127.0.0.1:8000/"
response1 = requests.get(url)
print(f"status get request: {response1.status_code}")

# url='https://geof-census-app.herokuapp.com/predict'
url = url + "predict"
r = requests.post(
    url=url,
    json=data,
)
print(f"status post request: {r.status_code}")
print(r.json())
