import json
import requests

# Send a GET using the URL http://127.0.0.1:8000
url = "http://127.0.0.1:8000"
r = requests.get(url)

# Print the status code
print("GET Status Code:", r.status_code)

# Print the welcome message
print("GET Response:", r.json())



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST using the data above
post_url = "http://127.0.0.1:8000/data/"
r = requests.post(post_url, json=data)

# Print the status code
print("POST Status Code:", r.status_code)

# Print the result
print("POST Result:", r.json())
