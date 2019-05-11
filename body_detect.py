import os
import json
import requests

# load keys from environ
api_key = os.environ["fpAPI"]
api_secret = os.environ["fpAPISecret"]

def get_body():
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    body = {
            "api_key": api_key,
            "api_secret": api_secret,
            "image_url": "https://bambiniphoto.sg/wp-content/uploads/family-photography-bambini-025.jpg",
            "return_attributes": 'gender,upper_body_cloth,lower_body_cloth'
            }
    
    print(body)

    try: # connection for POST
        req = requests.post('https://api-us.faceplusplus.com/humanbodypp/v1/detect', data=body, headers=headers)
        print(req.json())

    except Exception as e:
        print("Error")


get_body()