###THE CODE WHICH ALLOWS ME TO SEND INPUT TO THE GAME###

import ctypes
import time
import pyautogui

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


### HEX CODES FOR THE KEYS NEEDED
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

cam_left = 0x27
cam_right = 0x28
jump = 0x16



### MAKES SLIGHT ADJUSTMENTS IN THE 
### GIVEN DIRECTION FOR ALIGNING
def align_with_keys(direction):
    global A,S,W,D

    s_time = 0.06

    if(direction == "left"):

        PressKey(D)
        PressKey(S)
        time.sleep(s_time)
        ReleaseKey(D)
        ReleaseKey(S)

        PressKey(A)
        PressKey(W)
        time.sleep(s_time)
        ReleaseKey(A)
        ReleaseKey(W)
    else:

        PressKey(A)
        PressKey(S)
        time.sleep(s_time)
        ReleaseKey(A)
        ReleaseKey(S)

        PressKey(D)
        PressKey(W)
        time.sleep(s_time)
        ReleaseKey(D)
        ReleaseKey(W)



###IMPORTS FOR PYTORCH, IMAGE PROCESSING, AND OTHER HELPFUL OPERATIONS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import datasets, models, transforms
import time
import os
import os.path
import copy
from PIL import Image
import cv2
import pyautogui


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



###HELPER MODEL FOR CHOOSING SHORTEST ALIGNMENT DIRECTION 
#(e.g. left or right)
model_rotate = models.resnet18(pretrained=True)
num_ftrs = model_rotate.fc.in_features
model_rotate.fc = nn.Linear(num_ftrs, 2)
model_rotate = model_rotate.to(device)
criterion = nn.CrossEntropyLoss()
#Load the weights of the trained model
model_rotate.load_state_dict(torch.load('rotate.pth', map_location=torch.device('cuda')))
model_rotate.eval()
model_rotate.cuda()


###MODEL FOR ALIGNING WITH 0 DEGREES
model_0 = models.resnet18(pretrained=True)
num_ftrs = model_0.fc.in_features
model_0.fc = nn.Linear(num_ftrs, 2)
model_0 = model_0.to(device)
criterion = nn.CrossEntropyLoss()
#Load the weights of the trained model
model_0.load_state_dict(torch.load('align.pth', map_location=torch.device('cuda')))
model_0.eval()
model_0.cuda()


###MODEL FOR ALIGNING WITH 90 DEGREES
model_90 = models.resnet18(pretrained=True)
num_ftrs = model_90.fc.in_features
model_90.fc = nn.Linear(num_ftrs, 3)
model_90 = model_90.to(device)
criterion = nn.CrossEntropyLoss()
#Load the weights of the trained model
model_90.load_state_dict(torch.load('angle90.pth', map_location=torch.device('cuda')))
model_90.eval()
model_90.cuda()



###MODEL FOR ALIGNING WITH 270 DEGREES
model_270 = models.resnet18(pretrained=True)
num_ftrs = model_270.fc.in_features
model_270.fc = nn.Linear(num_ftrs, 3)
model_270 = model_270.to(device)
criterion = nn.CrossEntropyLoss()
#Load the weights of the trained model
model_270.load_state_dict(torch.load('angle270.pth', map_location=torch.device('cuda')))
model_270.eval()
model_270.cuda()



###FOR CONVERTING IMAGES TO TENSOR
preprocess = transforms.Compose([
        transforms.ToTensor()])




#predicts if it's shorter to turn left
#or right for alignment
def get_rotation_direction():

    global model_rotate
    global device
    global preprocess

    #capture the screen and crop the car region
    image_pil = pyautogui.screenshot().resize((640,360))
    image_pil = image_pil.crop((193, 209, 444, 350))
    frame = np.array(image_pil)
    frame = frame[:, :, ::-1].copy()

    #convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #create the hsv mask to filter only the car color
    lower_blue = np.array([36,122,130])
    upper_blue = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    

    #apply the mask
    result = cv2.bitwise_and(frame, frame, mask = mask) 

    #rearrange the channels for proper inference
    result = result[:,:, ::-1]

    #transform to tensor for Pytorch predicting
    image_pil = Image.fromarray(result)
    img_pil_preprocessed = preprocess(image_pil)
    batch_img_pil_tensor = torch.unsqueeze(img_pil_preprocessed, 0)

    #predict using the rotate model
    with torch.no_grad():
        prediction = model_rotate(batch_img_pil_tensor.to(device))
        
        _, pred = torch.max(prediction, 1)
        pred = int(pred[0])
        
        if(pred == 0):
            return "left"
        else:
            return "right"




#gets prediction for if car is aligned to 0 degrees
def get_aligned_0():
    

    global model_0
    global device
    global preprocess

    #capture the screen and crop the car region
    image_pil = pyautogui.screenshot().resize((640,360))
    image_pil = image_pil.crop((193, 209, 444, 350))
    frame = np.array(image_pil)
    frame = frame[:, :, ::-1].copy()

    #convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #create the hsv mask to filter only the car color
    lower_blue = np.array([36,122,130])
    upper_blue = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    

    #apply the mask
    result = cv2.bitwise_and(frame, frame, mask = mask) 

    #rearrange the channels for proper inference
    result = result[:,:, ::-1]

    #transform to tensor for Pytorch predicting
    image_pil = Image.fromarray(result)
    img_pil_preprocessed = preprocess(image_pil)
    batch_img_pil_tensor = torch.unsqueeze(img_pil_preprocessed, 0)

    #predict using the 0 model
    with torch.no_grad():
        prediction = model_0(batch_img_pil_tensor.to(device))
        
        _, pred = torch.max(prediction, 1)
        pred = int(pred[0])
        

        return pred


#gets prediction for if car is aligned to 90 degrees
def get_aligned_90():

    global model_90
    global device
    global preprocess

    #capture the screen and crop the car region
    image_pil = pyautogui.screenshot().resize((640,360))
    image_pil = image_pil.crop((193, 209, 444, 350))
    frame = np.array(image_pil)
    frame = frame[:, :, ::-1].copy()

    #convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #create the hsv mask to filter only the car color
    lower_blue = np.array([36,122,130])
    upper_blue = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    

    #apply the mask
    result = cv2.bitwise_and(frame, frame, mask = mask) 

    #rearrange the channels for proper inference
    result = result[:,:, ::-1]

    #transform to tensor for Pytorch predicting
    image_pil = Image.fromarray(result)
    img_pil_preprocessed = preprocess(image_pil)
    batch_img_pil_tensor = torch.unsqueeze(img_pil_preprocessed, 0)


    #predict using the 90 model
    with torch.no_grad():
        prediction = model_90(batch_img_pil_tensor.to(device))
        
        _, pred = torch.max(prediction, 1)
        pred = int(pred[0])
        
        return pred


#gets prediction for if car is aligned to 270 degrees
def get_aligned_270():

    global model_270
    global device
    global preprocess

    #capture the screen and crop the car region
    image_pil = pyautogui.screenshot().resize((640,360))
    image_pil = image_pil.crop((193, 209, 444, 350))
    frame = np.array(image_pil)
    frame = frame[:, :, ::-1].copy()

    #convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #create the hsv mask to filter only the car color
    lower_blue = np.array([36,122,130])
    upper_blue = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    

    #apply the mask
    result = cv2.bitwise_and(frame, frame, mask = mask) 

    #rearrange the channels for proper inference
    result = result[:,:, ::-1]

    #transform to tensor for Pytorch predicting
    image_pil = Image.fromarray(result)
    img_pil_preprocessed = preprocess(image_pil)
    batch_img_pil_tensor = torch.unsqueeze(img_pil_preprocessed, 0)

    #predict using the 270 model
    with torch.no_grad():
        prediction = model_270(batch_img_pil_tensor.to(device))
        
        _, pred = torch.max(prediction, 1)
        pred = int(pred[0])
        
        return pred





### CODE TO LOAD THE MASK R-CNN
def get_instance_segmentation_model_not_pretrained(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model



#Load the Mask R-CNN with 4 classes (background, ball, car, and goal)
model_mask = get_instance_segmentation_model_not_pretrained(4)
model_mask = model_mask.to(device)
#Load the pretrained model
model_mask.load_state_dict(torch.load('model_final_updatedgoal2.pth'))
model_mask.eval()




#Predicts using the Mask R-CNN
#returns the bounding boxes and
#the confidence scores for each 
#object
def get_MaskRCNN_pred_data():

    global device
    global model_mask

    #capture the screen and downsample to 640x360
    image_pil = pyautogui.screenshot().resize((640,360))
    
    #convert to tensor and predict
    convert_tensor = transforms.ToTensor()
    converted = convert_tensor(image_pil).unsqueeze(0).to(device)
    prediction = model_mask(converted)

    #get the boxes and labels
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    labels = prediction[0]['labels'].cpu().detach().numpy()

    has_ball = False
    has_car = False
    has_goal = False

    #check which objects are present if any
    if(len(prediction[0]['boxes']) > 1 and len(np.unique(labels))>1):
            ballindex = np.where((labels == 1))[0]
            carindex = np.where((labels == 2))[0]
            goalindex = np.where((labels == 3))[0]

            if(len(ballindex) > 0):
                has_ball = True
                ballindex = ballindex[0]
            if(len(carindex) > 0):
                has_car = True
                carindex = carindex[0]
            if(len(goalindex) > 0):
                has_goal = True
                goalindex = goalindex[0]

    preds_ball = []
    preds_car = []
    preds_goal = []

    score_ball = 0
    score_car = 0
    score_goal = 0

    #get the boxes and scores of objects that were present
    if(has_ball):
        preds_ball = boxes[ballindex]
        score_ball = prediction[0]['scores'][ballindex].cpu().detach().numpy()

    if(has_car):
        preds_car = boxes[carindex]
        score_car = prediction[0]['scores'][carindex].cpu().detach().numpy()

    if(has_goal):
        preds_goal = boxes[goalindex]
        score_goal = prediction[0]['scores'][goalindex].cpu().detach().numpy()

    return [(preds_ball, score_ball), (preds_car, score_car), (preds_goal, score_goal) ]



#gets the distance from the car to the 
#ball using:
#distance = top y coord car box - bottom y coord ball box
def get_car_to_ball_distance():

    #get the Mask R-CNN prediction
    data = get_MaskRCNN_pred_data()

    #get the boxes for car and ball
    preds_ball = data[0][0]
    preds_car = data[1][0]

    #make sure the data exists
    if(len(preds_ball) != 0 and len(preds_car) != 0):
        
        #get the scores to make sure the predictions 
        #are confident enough to trust
        score_ball = data[0][1]
        score_car = data[1][1]

        #only trust predictions above certain confidence
        if(score_ball > 0.7 and score_car > 0.7):

            #calculate distance
            distance = preds_car[1]-preds_ball[3]

            return distance
        else:
            return None


#checks to see if the goal is to the left
#or right of the car, which helps the 
#car figure out which direction to circle
def check_goal_is_left_or_right():

    #pan camera left if necessary
    PressKey(cam_left)
    time.sleep(1)

    #get the data of panned view
    data = get_MaskRCNN_pred_data()
    preds_car = data[1][0]
    score_car = data[1][1]
    preds_goal = data[2][0]
    score_goal = data[2][1]
    ReleaseKey(cam_left)

    #if the goal was there with high confidence
    #return left
    if(len(preds_goal) != 0 and score_goal > 0.9):
        return "left"

    time.sleep(1)

    #pan camera right if necessary
    PressKey(cam_right)
    time.sleep(1)
    #get the data of panned view
    data = get_MaskRCNN_pred_data()
    preds_goal = data[2][0]
    score_goal = data[2][1]
    ReleaseKey(cam_right)

    #if the goal was there with high confidence
    #return right
    if(len(preds_goal) != 0 and score_goal > 0.9):
        return "right"


    return None


#uses the Mask R-CNN prediction to 
#see if the ball is between the goal
#posts by checking bounding boxes
def ball_aligned_in_goal_center():

    #get the data
    data = get_MaskRCNN_pred_data()

    #get the scores and boxes
    preds_ball = data[0][0]
    score_ball = data[0][1]
    preds_goal = data[2][0]
    score_goal = data[2][1]

    

    #if the ball or goal can't be seen return false
    if(len(preds_goal) == 0 or len(preds_ball) == 0):
        return False

    #if the ball is inside the goal bounding box with high confidence
    #return true
    if(preds_ball[0] > preds_goal[0] and preds_ball[2] < preds_goal[2]
        and score_goal > 0.7 and score_ball > 0.7):
        return True
    else:
        return False




###LOGIC TO SCORE GOAL STARTS HERE

#Give small sleep time just for swapping from terminal to game
time.sleep(2)
print("starting to score")


#initially the goal direction is unknown
goal_direction = "unknown"


#get the bounding box and score data
data = get_MaskRCNN_pred_data()

car_preds = data[1][0] 
score_car = data[1][1]
goal_preds = data[2][0]
score_goal = data[2][1]

#check to see if the goal exists and if the direction 
#can be determined, keep track if it can since this
#won't change as the car moves closer
if(len(goal_preds) != 0 and len(car_preds) != 0 and 
    score_car > 0.7 and score_goal > 0.7):
    
    if(goal_preds[2] > car_preds[2]):
        goal_direction = "right"
    elif(goal_preds[0] < car_preds[0]):
        goal_direction = "left"


#get the best direction to turn for 
#aligning with the ball
dir_to_rotate = get_rotation_direction()

#align with the ball
while(get_aligned_0() == 1):
    align_with_keys(dir_to_rotate)
    

#get current distance to ball
distance = get_car_to_ball_distance()


#move the car to a distance of 40
#from the ball
while(distance >= 40):

    PressKey(W)
    
    distance = get_car_to_ball_distance()

    #if the ball and car are too close this can happen
    #so just break because the car is close enough
    if(distance == None):
        break


#to stop forward momentum press back quickly
ReleaseKey(W)
PressKey(S)
time.sleep(0.3)
ReleaseKey(S)



#if it undershoots move forward a little
while(distance > 33):
    PressKey(W)
    time.sleep(0.25)
    ReleaseKey(W)
    distance = get_car_to_ball_distance()

#if it overshoots move backwards a little
while(distance < 27):
    PressKey(S)
    time.sleep(0.25)
    ReleaseKey(S)
    distance = get_car_to_ball_distance()



#if earlier it couldn't deduce where the goal was
#pan the camera so it can find it 
if(goal_direction == "unknown"):

    goal_direction = check_goal_is_left_or_right()

    #break execution if for some reason goal isn't
    #visible (hasn't occurred ever since camera can
    #pan the whole field basically)
    if(goal_direction == None):
        print("ERROR CAN'T FIND GOAL")
        1/0


#circle to the left if the goal was to the right
if(goal_direction == "right"):


    #align to 270 degrees
    PressKey(A)
    while(get_aligned_270() != 0):
        PressKey(W)
        time.sleep(0.06)
        ReleaseKey(W)
    ReleaseKey(A)

    #circle until the ball is in the center of the 
    #goal
    while(ball_aligned_in_goal_center() == False):

        #figure out if the car can move forward
        #or needs to realign to 270
        direction_to_adjust = get_aligned_270()

        if(direction_to_adjust == 0):
            PressKey(W)
            time.sleep(0.25)
            ReleaseKey(W)
        elif(direction_to_adjust == 2):
            PressKey(A)
            for i in range(1):
                PressKey(W)
                time.sleep(0.25)
                ReleaseKey(W)
            ReleaseKey(A)
        else:
            PressKey(D)
            for i in range(1):
                PressKey(W)
                time.sleep(0.25)
                ReleaseKey(W)
            ReleaseKey(D)

#circle to the right if the goal was to the left
else:

    #align to 90 degrees
    PressKey(D)
    while(get_aligned_90() != 0):

        PressKey(W)
        time.sleep(0.06)
        ReleaseKey(W)

    ReleaseKey(D)

    #circle until the ball is in the center of the 
    #goal
    while(ball_aligned_in_goal_center() == False):


        #figure out if the car can move forward
        #or needs to realign to 90
        direction_to_adjust = get_aligned_90()


        if(direction_to_adjust == 0):
            PressKey(W)
            time.sleep(0.25)
            ReleaseKey(W)
        elif(direction_to_adjust == 2):
            PressKey(A)
            for i in range(1):
                PressKey(W)
                time.sleep(0.25)
                ReleaseKey(W)
            ReleaseKey(A)
        else:
            PressKey(D)
            for i in range(1):
                PressKey(W)
                time.sleep(0.25)
                ReleaseKey(W)
            ReleaseKey(D)






#now the ball is between the goal posts
#the car just needs to align itself

#get shortest direction to align with ball
dir_to_rotate = get_rotation_direction()


#align with the ball
while(get_aligned_0() == 1):
    align_with_keys(dir_to_rotate)




#move slightly closer if it circled 
#too far
while(distance > 30):
    PressKey(W)
    time.sleep(0.25)
    ReleaseKey(W)
    distance = get_car_to_ball_distance()


#make a slight adjustment if necessary
while(get_aligned_0() == 1):
    align_with_keys(dir_to_rotate)


#get distance to ball, this helps determine
#timing of any last minute turns
distance = get_car_to_ball_distance()


#move forward to take the shot
PressKey(W)

#sleep based on distance 
#which helps make one final 
#inference check at roughly the 
#same spot
time.sleep(0.6*(distance/40))


#make one final inference on bounding
#box data to see if a slight left or right
#turn needs to be made
data = get_MaskRCNN_pred_data()

preds_car = data[1][0]
preds_ball = data[0][0]
score_ball = data[0][1]
preds_goal = data[2][0]
score_goal = data[2][1]




#calculate some bounding box distances to see if a rotation needs
#to be made
car_right_minus_ball_right = preds_car[2] - preds_ball[2]
ball_left_minus_car_left = preds_ball[0] - preds_car[0]
distance = preds_car[3]-preds_ball[1]



#if a slight right turn should be made
if(ball_left_minus_car_left < car_right_minus_ball_right and 
    (car_right_minus_ball_right - ball_left_minus_car_left ) >= 3):
    PressKey(D)
    time.sleep(0.25*(distance/28))
    ReleaseKey(D)

#if a slight left turn should be made
if(ball_left_minus_car_left > car_right_minus_ball_right and 
    (ball_left_minus_car_left - car_right_minus_ball_right) >= 3):
    PressKey(A)
    time.sleep(0.25*distance/28)
    ReleaseKey(A)




#keep moving forward after any slight adjustments to finish
#the shot
for i in range(4):
    time.sleep(0.5)



ReleaseKey(W)









