import os
import json
import requests
import operator

# load keys from environ
api_key = os.environ["fpAPI"]
api_secret = os.environ["fpAPISecret"]

def match_faces_to_bodies(faces, bodies):
    for face in faces:
        faceLocation = face['face_rectangle']
    return 0

def get_faces(filemame='./frames/shopCouple.jpg'):
    # gender
    # age
    # smiling
    # headpose
    # facequality
    # blur
    # eyestatus
    # emotion
    # ethnicity
    # beauty
    # mouthstatus
    # eyegaze
    # skinstatus
    attributes = "gender,age,smiling,emotion,ethnicity,beauty"

    files = {
        'api_key': (None, api_key),
        'api_secret': (None, api_secret),
        'image_file': ('image_file.jpg', open('./frames/shopCouple.jpg', 'rb')),
        'return_attributes': (None, attributes),
    }

    try: # connection for POST
        req = requests.post('https://api-us.faceplusplus.com/facepp/v3/detect', files=files)

        return json.loads(req.text)

    except Exception:
        print("Error")

def get_bodies(filemame='./frames/shopCouple.jpg'):

    files = {
        'api_key': (None, api_key),
        'api_secret': (None, api_secret),
        'image_file': ('image_file.jpg', open(filename, 'rb')),
        'return_attributes': (None, 'gender,upper_body_cloth,lower_body_cloth'),
    }


    try: # connection for POST
        req = requests.post('https://api-us.faceplusplus.com/humanbodypp/v1/detect', files=files)

        return json.loads(req.text)

    except Exception:
        print("Error")


faces = get_faces()
people = get_bodies()

# math the faces and bodies
match_faces_to_bodies(faces['faces'], people['humanbodies'])

# NOTE that for upper / lower body colors we also have the option to grab RGB
# for human in people['humanbodies']:
#     sex = human['attributes']['gender']
#     upper_body_cloth_color = human['attributes']['upper_body_cloth']#['upper_body_cloth_color']
#     lower_body_cloth_color = human['attributes']['lower_body_cloth']#['lower_body_cloth_color']
#     rectangle = human['humanbody_rectangle']
#     #print(f"rectangle: {rectangle}, sex: {sex}, upper: {upper_body_cloth_color}, lower: {lower_body_cloth_color}")