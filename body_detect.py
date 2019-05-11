import os
import json
import requests
import operator
import math
from scipy.spatial.distance import euclidean

## SET an environment script with your credentials
# eg.
# #!/bin/bash
# export fpAPI=<yourAPI>
# export fpAPISecret=<yourSecret>

# THEN activate the environment
# source <yourscript>.sh

# load keys from environment
api_key = os.environ["fpAPI"]
api_secret = os.environ["fpAPISecret"]

# Represents a customer
class Customer(object):
    # the time series state of the person
    activity_record = []
    
    def __init__(self, person):
        self.activity_record.append(person)

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

            if dist < minDistance:
                minDistance = dist
                minIdx = idx

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
        'image_file': ('image_file.jpg', open(filename, 'rb')),
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

# Create a faceset for the PARAM: camera
# facesets can contain 'tags' which can then be queried to return all faceset with that tag
def create_faceSet(tags):

    files = {
        'api_key': (None, api_key),
        'api_secret': (None, api_secret),
        'tag': (None, 'person,'+tags),
    }

    try: # connection for POST
        req = requests.post('https://api-us.faceplusplus.com/facepp/v3/faceset/create', files=files)
        return json.loads(req.text)

    except Exception:
        print("Error")


# Keeps track of faces in the Store across ALL cameras
# used for TIME spent shopping PER customer as well as demographic data via Tags
#
# called from get_faces
# i) check if face exists -> compare API (tokens against FACESET)
# ii) if not exist ADD new token
def update_faceSet(faces):
    # get facesets for comparison
    files = {
    'api_key': (None, '<api_key>'),
    'api_secret': (None, '<api_secret>'),
    }

    req = requests.post('https://api-us.faceplusplus.com/facepp/v3/faceset/getfacesets', files=files)
    facesets = json.loads(req.text)

    for face in faces['faces']:
        files = {
            'api_key': (None, '<api_key>'),
            'api_secret': (None, '<api_secret>'),
            'face_token1': (None, face['face_token']),
            'face_token2': (None, 'ad248a809408b6320485ab4de13fe6a9'),
        }

        response = requests.post('https://api-us.faceplusplus.com/facepp/v3/compare', files=files)



#============ Program Start

totalNumberCustomers = None
facesetTokens = {}

# create facesets to individualy track males and females
facesetTokens['male'] = create_faceSet("male")["faceset_token"]
facesetTokens['female'] = create_faceSet("female")["faceset_token"]
print(facesetTokens)

# NOTE: we do not need to update at EVERY frame. To save time we will call updates every X
# seconds and then use the intermediate time to process the result with Compare API etc.
faces = get_faces()
bodies = get_bodies()

# Update totalNumberCustomers
totalNumberCustomers = len(bodies['humanbodies'])
print(totalNumberCustomers)

# match the faces and bodies
people = match_faces_to_bodies(faces['faces'], bodies['humanbodies'])
#print(people)