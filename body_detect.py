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

# GLOBALS ================================
globalTimeStep = 0 # track updates by timestep
customers = [] # list of Customer Objects
totalNumberCustomers = None

# Represents a customer
class Customer(object):
    
    def __init__(self, person, entry_time):
        self.activity_record = [person] # customer first detected
        self.entry_time = entry_time # customer state last updated
        self.last_update_time = entry_time # considered exited after X timesteps of no update
        self.exit_time = None

    def __repr__(self):
        return(f"entry {self.entry_time}, last_update {self.last_update_time}, exit {self.exit_time}, {self.activity_record}")



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
def get_faces(filename):
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
def get_bodies(filename):

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

# Process people takes all the new identified people (body/face pair) in the last frame
# and will see if (based on position) this person is NEW or part of the last frame
# IF new, then it will create a new customer record
# ELSE, will add the persons current tuple state to the existing customer record.
def process_people(people):
    global customers
    # Initially there are NO customers so we take all in the frame
    if globalTimeStep <= 1:
        for person in people:
            customers.append(Customer(person, globalTimeStep))

    # # Filter customers to process (will only look for customers who were updates in that last timestep)
    # customers_prevFrame = [cust for customers if cust.last_update_time == globalTimeStep-1]

# NOTE: we do not need to update at EVERY frame. To save time we will call updates every X
# seconds and then use the intermediate time to process the result with Compare API etc.
def processFrame(filename='./frames/shopCouple.jpg'):
    global globalTimeStep

    print(filename)
    globalTimeStep += 1
    faces = get_faces(filename)
    print(faces)
    bodies = get_bodies(filename)
    print(bodies)

    # Update totalNumberCustomers
    totalNumberCustomers = len(bodies['humanbodies'])
    #print(totalNumberCustomers)

    # match the faces and bodies
    people = match_faces_to_bodies(faces['faces'], bodies['humanbodies'])
    #print(people)
    #print("PEOPLE------------------")
    process_people(people)

#============ Program Start

# create facesets to individualy track males and females
# facesetTokens['male'] = create_faceSet("male")["faceset_token"]
# facesetTokens['female'] = create_faceSet("female")["faceset_token"]

startFrame = 1
numFrames = 9
for i in range(startFrame,startFrame+numFrames):
    # padded to filenames of data set
    #processFrame("./Retail/women01/resized/"+str(i).rjust(5,"0")+".jpg")
    processFrame("./products/"+str(i)+".png")
