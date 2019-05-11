import os
import json
import requests
import operator

# load keys from environ
api_key = os.environ["fpAPI"]
api_secret = os.environ["fpAPISecret"]

def get_bodies():

    # files={'image_file': open("./frames/family-photography-bambini-025.jpg", 'rb')}
    # headers = {'content-type': 'application/x-www-form-urlencoded'}
    # body = {
    #         "api_key": api_key,
    #         "api_secret": api_secret,
    #         "return_attributes": 'gender,upper_body_cloth,lower_body_cloth'
    #         "image_file": files['image_file']
    #         }
    # #print(body)

    files = {
        'api_key': (None, api_key),
        'api_secret': (None, api_secret),
        'image_file': ('image_file.jpg', open('./frames/family-photography-bambini-025.jpg', 'rb')),
        'return_attributes': (None, 'gender,upper_body_cloth,lower_body_cloth'),
    }


    try: # connection for POST
        req = requests.post('https://api-us.faceplusplus.com/humanbodypp/v1/detect', files=files)
        print(req.json())

        return json.loads(req.text)

    except Exception as e:
        print("Error")


people = get_bodies()
print(people)

# for human in people['humanbodies']:
#     sex = human['attributes']['gender']
#     sex = 
#     upper_body_cloth_color = human['attributes']['upper_body_cloth_color']
#     lower_body_cloth_color = human['attributes']['lower_body_cloth_color']

#     print(f"sex: {sex}, upper: {upper_body_cloth_color}, lower: {lower_body_cloth_color}")