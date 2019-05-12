from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import os
import json
import requests
import operator
import math
import base64
from scipy.spatial.distance import euclidean
from pprint import pprint

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import pickle
import numpy as np
import os
import random

import sklearn
import traceback, sys, os 

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import plot_image, plot_keypoints


import threading
import time




from PIL import Image
from mxnet import image as img
from io import BytesIO

parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--detector', type=str, default='yolo3_mobilenet1.0_coco',
                    help='name of the detection model to use')
parser.add_argument('--pose-model', type=str, default='simple_pose_resnet50_v1b',
                    help='name of the pose estimation model to use')
parser.add_argument('--num-frames', type=int, default=1000,
                    help='Number of frames to capture')
opt = parser.parse_args()




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

    if len(people)==0:
      people.append((None,bodies[0]))

    return people

def write(img_arr, flag=1, output_format='jpeg', dim_order='HWC'):
    """
    Write an NDArray to a base64 string.
    :param img_arr: NDArray
        Image in NDArray format with shape (channel, width, height).
    :param flag: {0, 1}, default 1
        1 for three channel color output. 0 for grayscale output.
    :param output_format: str
        Output image format.
    :param dim_order: str
        Input image dimension order. Valid values are 'CHW' and 'HWC'
    :return: str
        Image in base64 string format
    """

    #print(img_arr.size)
    assert dim_order in 'CHW' or dim_order in 'HWC', "dim_order must be 'CHW' or 'HWC'."
    if dim_order == 'CHW':
        img_arr = mx.nd.transpose(img_arr, (1, 2, 0))
    if flag == 1:
        mode = 'RGB'
    else:
        mode = 'L'
        img_arr = mx.nd.reshape(img_arr, (img_arr.shape[0], img_arr.shape[1]))
    img_arr = img_arr.astype(np.uint8).asnumpy()
    image = Image.fromarray(img_arr, mode)
    output = BytesIO()
    image.save(output, format=output_format)
    output.seek(0)
    if sys.version_info[0] < 3:
        return base64.b64encode(output.getvalue())
    else:
        return base64.b64encode(output.getvalue()).decode("utf-8")

def encodeBase64Image(image):

  #encoded_string = base64.b64encode(image)
  #image = mx.img#.imdecode(image)

  transformer = transforms.Resize(size=(128, 128),keep_ratio=True)
  #image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
  img2 = transformer(image)

  encoded_string = write(img2)
  return encoded_string

# Calls API to return features of faces in the frame
def get_faces_(image):
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

    data = {
        'api_key': (None, api_key),
        'api_secret': (None, api_secret),
        'image_base64': encodeBase64Image(image),
        'return_attributes': (None, attributes),
    }

    try: # connection for POST
        req = requests.post('https://api-us.faceplusplus.com/facepp/v3/detect', data=data)
        return json.loads(req.text)

    except Exception:
        print("Error")

# Calls API to return features of bodies in the frame
def get_bodies_(image):

    data = {
        'api_key': (None, api_key),
        'api_secret': (None, api_secret),
        'image_base64': encodeBase64Image(image),
        'return_attributes': (None, 'gender,upper_body_cloth,lower_body_cloth'),
    }


    try: # connection for POST
        req = requests.post('https://api-us.faceplusplus.com/humanbodypp/v1/detect', data=data)
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
# used for TIME spent shopping PER customer
# used for total COUNT of all customers in store
#
# called from get_faces
# i) check if face exists -> compare API (tokens against FACESET)
# ii) if not exist ADD new token
def update_faceSet(faces):
    for face in faces:
        files = {
            'api_key': (None, '<api_key>'),
            'api_secret': (None, '<api_secret>'),
            'face_token1': (None, 'c2fc0ad7c8da3af5a34b9c70ff764da0'),
            'face_token2': (None, 'ad248a809408b6320485ab4de13fe6a9'),
        }

        response = requests.post('https://api-us.faceplusplus.com/facepp/v3/compare', files=files)

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

    #pprint(customers)
    # # Filter customers to process (will only look for customers who were updates in that last timestep)
    # customers_prevFrame = [cust for customers if cust.last_update_time == globalTimeStep-1]

    # # iterate through all the new people in frame
    # for person in people:
    #     body = person[1] # persons body
    #     body_coords = body['humanbody_rectangle']
    #     print(body['humanbody_rectangle'])
        
    #     check euclidean to people in 
    #     for cust in customers:
    #return 0



# NOTE: we do not need to update at EVERY frame. To save time we will call updates every X
# seconds and then use the intermediate time to process the result with Compare API etc.
def processFrame(image):
    global globalTimeStep
    global customers

    customers = []

    globalTimeStep += 1
    faces = get_faces_(image)
    #print(faces[list(faces.keys())[1]])
    #print()
    bodies = get_bodies_(image)
    if bodies is None:
      bodies = {}
    if faces is None:
      faces = {}
    #print(bodies[list(faces.keys())[1]])
    #print(bodies.keys())
    #print()

    # print(bodies.keys())
    # print(faces.keys())
    # try:
    #   print(faces['humanbodies'])
    # except:
    #   pass
    # print()

    # Update totalNumberCustomers
    if 'humanbodies' in bodies.keys() and 'faces' in faces.keys():
      totalNumberCustomers = len(bodies['humanbodies'])
      print(totalNumberCustomers)

      # match the faces and bodies
      people = match_faces_to_bodies(faces['faces'], bodies['humanbodies'])
      #print(people)
      #print("PEOPLE------------------")
      #process_people(people)
      return people

    return []


#===========plugin api

def writeToDb(dbrecord):

  with open('trackingCustomer.db','a') as f:

    f.write(dbrecord)

def facepp_plugin(image,action):

    global api_ready
    api_ready = False
    people = processFrame(image)
    writeToDb("frame processed! with {} \n".format(len(people)))
    api_ready = True

    if people is None or len(people)==0:
      return

    # print intersting stuff
    for customer in people:
      
      face = customer[0] 
      body = customer[1]

      if body is not None:

        body = customer[1]['attributes']
        outfit = str(body['lower_body_cloth']['lower_body_cloth_color']) + \
           "_" + str(body['upper_body_cloth']['upper_body_cloth_color']) 
      else:
        outfit = "nude"

      if face is not None:
        face = customer[0]['attributes']
        ethnicity = face['ethnicity']['value']
        gender = face['gender']['value']
        emotions = [ v for k,v in face['emotion'].items()]
        emotions_score = [ v for k,v in face['emotion'].items()]
        emotion_max = max(emotions_score)
        emotion_i = emotions_score.index(emotion_max)
        emotion = "coldice"
        try:
          emotion = face['emotion'][emotions[emotion_i]]
        except:
          pass
      else:
        ethnicity = "unknown"
        outfit = "unknown"
        emotion = "unknown"
        gender = "angel"

      dbrecord = ethnicity + " "+ gender + " " + emotion + " " + outfit + " action: "+ str(action) +"\n"
      print(dbrecord)
      writeToDb(dbrecord)

    

def facepp_plugin_setup():
    #============ Program Start

    # create facesets to individualy track males and females
    global facesetTokens
    global globalTimeStep 
    global customers 
    global totalNumberCustomers
    # load keys from environment
    global api_key
    global api_secret

    api_key = os.environ["fpAPI"]
    api_secret = os.environ["fpAPISecret"]

    # GLOBALS ================================
    globalTimeStep = 0 # track updates by timestep
    customers = [] # list of Customer Objects
    totalNumberCustomers = None
    
    

# arms raised function
def actionArmsRaised(pred_coords, upscale_bbox):
    # width
    x = pred_coords[0,:,0].asnumpy()
    y = pred_coords[0,:,1].asnumpy()
    width = max(x) - min(x)
    height = max(y) - min(y)



    minx = max(x)
    maxx = min(x)
    miny= float(min(y))
    maxy = float(max(y))
    
    
    # centroid of body skeleton
    # xavg in the middle of 2 shoulders
    #shoulders = pred_coords[0,-7:-5,0].asnumpy()
    xavg = x.sum()/len(x)
    yavg = y.sum()/len(y)

    # rest of the computation with what we believe are arms only
    #x = pred_coords[0,5:-6,0].asnumpy()
    #y = pred_coords[0,5:-6,1].asnumpy()

    #print("shoulders",shoulders)
    #print("center",xavg, yavg)
    #print() 

    #print("arms",x,y)
    
    # variance and standard deviation
    xvar = (x - xavg)**2
    xvar = xvar.sum()/(len(xvar) - 1)
    xsd = np.sqrt(xvar)
    yvar = (y - yavg)**2
    yvar = yvar.sum()/(len(yvar) - 1)
    ysd = np.sqrt(yvar)
    

    # goal detect raised arms
    # IDEA1) extreme nodes: more than 1sd away from it
    d_xG = x - xavg
    #print(abs(d_xG))
    #print(abs(d_xG) - xsd)
    #print(abs(d_xG) - 2*xsd)
    x_extr_1sd = abs(d_xG) - 2.5*xsd > 0 
    x_extr_2sd = abs(d_xG) - 3*xsd > 0 
    #print("x_extr_1sd: ", x_extr_1sd)
    #print("x_extr_2sd: ", x_extr_2sd)
    #print()
    # problem: var depends on the spread itself -> bad for detecting arm raised
    # sol: 2 thesholds 1sd and 2sd, if more than 10% in 2sd -> then it's not an arm raised
    # IDEA1.1) expect 2 nodes to be in 1sd and 1 node in 2sd
    # select those extreme nodes: in x_extr_1sd and 
    selectedx = x[x_extr_1sd]
    #print("selectedx ",selectedx)

    arms_raised_fact1 = x_extr_1sd.sum() >0
    #print("arms_rased_fact1: ",arms_raised_fact1)

    #IDEA 1.4) verify if order of nodes corresponds to parts of body
    #          then use it to compute values for arms only! no shoulders not legs



    # IDEA1.2) height -> extreme nodes have close height? -> arm raised
    # selected nodes are close in height, alltogether
    selectedy = y[x_extr_1sd]
    #print("selectedy ",selectedy)
    #d_yG = selectedy - yavg
    arms_raised_fact2 = False
    if len(selectedy)>0:
      maxy_extr = max(selectedy)
      miny_extr = min(selectedy)
      d_yextr = maxy_extr - miny_extr
      #y_extr_nodes = abs(d_yG) - ysd > 0 
      #y_extr2_nodes = abs(d_yG) - 2*ysd > 0 
      # height difference < 1sd height < halfwidth
      #arms_raised_fact2 =  d_yextr < ysd #and d_yextr < height/2
      #arms_raised_fact2 =  d_yextr < height*0.6
      arms_raised_fact2 = miny_extr > miny + height/2
      #print("ysd: ",ysd," d_yextr: ",d_yextr, " hieght/2: ", height*0.6)
      #print("arms_rased_fact2: ",arms_raised_fact2)
      # this fact is not realistic. forget it

    # IDEA1.3) verify bbox width is < height so it's a person standing and not bending
    arms_raised_fact3 = height > width # bounding box width is < height
    # PENDING: use upsale_bbox also
    
    #print(" Function result: ", arms_raised_fact1, arms_raised_fact3, selectedx, selectedy)
    arms_raised_action = arms_raised_fact1  and arms_raised_fact3
    
    return arms_raised_action, selectedx, selectedy
    




def getPose(im_fname):
  x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
  #print('Shape of pre-processed image:', x.shape)

  class_IDs, scores, bounding_boxs = detector(x)
  pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

  predicted_heatmap = pose_net(pose_input)
  pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

  return img, pred_coords, confidence, class_IDs, bounding_boxs, scores, upscale_bbox


def plotPose(img, pred_coords, confidence, class_IDs, bounding_boxs, scores):
  ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                                class_IDs, bounding_boxs, scores,
                                box_thresh=0.5, keypoint_thresh=0.2)
  plt.show()
  

def parseImage(imagepath):
  img, pred_coords, confidence, class_IDs, bounding_boxs, scores, upscale_bbox = getPose(imagepath)
  #plotPose(img, pred_coords, confidence, class_IDs, bounding_boxs, scores)
  

  #print(confidence)
  #print(scores[0,0])
  #print(bounding_boxs[0,0])
  #print(class_IDs)
  # select best score pose
  
  
  pred_coords = pred_coords.asnumpy()
  #print(pred_coords.shape)
  #print(pred_coords[0])
  return pred_coords[0]

# # arms raised function
# def actionArmsRaisedSVM(pred_coords, upscale_bbox):
   
#   # transform input data
#   global loaded_model
  

    
#   poses_np =[ pred_coords.flatten()]
    
#   #print(poses_np)
  
#   #result = loaded_model.predict(poses_np)
#   #print(result)
#   return result == 1
    



def hackathon_action(i,image, pred_coords, class_IDs, bounding_boxs, scores, box_thresh=0.5):
    """ 
        Infer action from those coords
        Steps:
            centroid
            total height
            total width
            detect

    """

    # try:
    #     detected_action2 = actionArmsRaisedSVM(pred_coords, bounding_boxs)
    #     if detected_action2:
    #         print(" Arms raised by SVM !")
    #         print()
    # except  Exception:
    #   #print("Exception in user code:")
    #   #print("-"*60)
    #   #traceback.print_exc(file=sys.stdout)
    #   #print("-"*60)
    #   pass


    #print(pred_coords, class_IDs, bounding_boxs, scores, "\n")
    detected_action, x, y = actionArmsRaised(pred_coords, bounding_boxs)
    if detected_action:
        print(" Touching at ({}, {}) !".format(x[0],y[0]))
        pprint(pred_coords)
        print()

    

    try:
      global pause_time
      global api_ready
      print(api_ready)
      if i % int(50) == 0 and api_ready:
        # attributes and outfit
        x = threading.Thread(target=facepp_plugin, args=(image,detected_action))
        x.start()
        
    except  Exception:
      #print("Exception in user code:")
      #print("-"*60)
      traceback.print_exc(file=sys.stdout)
      pass

    


    






def keypoint_detection(i,frame, detector, pose_net, ctx=mx.cpu(), axes=None):
    
    global pause_time
    

    x, img = gcv.data.transforms.presets.yolo.transform_test(frame, short=512, max_size=350)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    plt.cla()
    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs,
                                                       output_shape=(128, 96), ctx=ctx)

    #print(pose_input,"\n")
    if len(upscale_bbox) > 0 :

        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)



        hackathon_action(
            i,
            frame,
            pred_coords, 
            confidence, 
            class_IDs, 
            bounding_boxs, 
            scores)


        axes = plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2, ax=axes)
        plt.draw()
        plt.pause(pause_time)
        #plt.pause(1.0)
    else:
        axes = plot_image(frame, ax=axes)
        plt.draw()
        plt.pause(pause_time)

    return axes

if __name__ == '__main__':


    # facepp
    facepp_plugin_setup()

    # action recognition model
    global loaded_model

    global api_ready
    api_ready = True

    global pause_time
    pause_time = 0.001 #3.0 #0.001
    #filename = 'action_recognition_svm_local.sav'
    #loaded_model = pickle.load(open(filename, 'rb'))

    # webcam loop
    ctx = mx.cpu()
    detector_name = "ssd_512_mobilenet1.0_coco"
    detector = get_model(detector_name, pretrained=True, ctx=ctx)
    detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
    net = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)

    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 48)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)
    time.sleep(1)  ### letting the camera autofocus
    axes = None

    for i in range(opt.num_frames):


        ret, frame = cap.read()

        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        axes = keypoint_detection(i,frame, detector, net, ctx, axes=axes)
