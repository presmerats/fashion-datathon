import os
import json
import requests
import operator
import math
from scipy.spatial.distance import euclidean

# load keys from environ
api_key = os.environ["fpAPI"]
api_secret = os.environ["fpAPISecret"]

# Will match a face to a body using location of bounding box.
# each face will be contained in a bounding of a BODY.
# if face x,y in body x to x+width, y to y+height THEN it's a same person
#
# RETURNS a LIST of People (face, body)
def match_faces_to_bodies(faces, bodies):
    # key/value store of body to face
    people = []

    for face in faces:
        faceLocation = face['face_rectangle']

        minDistance = float('infinity')
        minIdx = None

        # iterate through all bodies and choose which is bounding the face best
        for idx, body in enumerate(bodies):
            bodyLocation = body['humanbody_rectangle']
            dist = euclidean([faceLocation["top"],faceLocation['left']], [bodyLocation['top'], bodyLocation["left"]]) 
            print(f"DIST ==== {dist}")
            if dist < minDistance:
                minDistance = dist
                minIdx = idx

        print(minIdx)
        # add face/body store (IDS)
        people.append((face,bodies[minIdx]))
        # remove the matched body from body LIST
        del bodies[minIdx]

    return people

# Calls API to return features of faces in the frame
def get_faces(filename='./frames/shopCouple.jpg'):
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

# Calls API to return features of bodies in the frame
def get_bodies(filename='./frames/shopCouple.jpg'):

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
bodies = get_bodies()

# math the faces and bodies
people = match_faces_to_bodies(faces['faces'], bodies['humanbodies'])
print(people)

# NOTE that for upper / lower body colors we also have the option to grab RGB
# for human in people['humanbodies']:
#     sex = human['attributes']['gender']
#     upper_body_cloth_color = human['attributes']['upper_body_cloth']#['upper_body_cloth_color']
#     lower_body_cloth_color = human['attributes']['lower_body_cloth']#['lower_body_cloth_color']
#     rectangle = human['humanbody_rectangle']
#     #print(f"rectangle: {rectangle}, sex: {sex}, upper: {upper_body_cloth_color}, lower: {lower_body_cloth_color}")