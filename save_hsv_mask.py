

import numpy as np
import os
import os.path
from PIL import Image
import cv2
import pyautogui
import keyboard




#directory to save to, change the angle number
#based on which class to focus on 
DIR = 'C://Users/laika/Desktop/PlayGames/angle_data/270/'
#count the files in the directory to get a unique name for each file
count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])




def save_cropped_and_color_masked():
	

	#used for unique filename
	global count

	#take a screenshot resize it and crop it
	image_pil = pyautogui.screenshot().resize((640,360))
	image_pil = image_pil.crop((193, 209, 444, 350)) 
	image_pil_copy = image_pil.copy()
	
	#convert to numpy and flip so that it's 
	#in the right format for cv2
	frame = np.array(image_pil)
	frame = frame[:, :, ::-1].copy()

	#convert from RGB to HSV 
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#define the lower and upper bounds to filter the blue
	#of the car
	lower_blue = np.array([36,122,130])
	upper_blue = np.array([100, 255, 255])
	#create the mask which will be applied
	mask = cv2.inRange(hsv, lower_blue, upper_blue)


	#apply the mask
	result = cv2.bitwise_and(frame, frame, mask = mask)


	cv2.imwrite(DIR + "frame" + str(count) + ".jpg", result) 

	count += 1

	# result = result[:,:, ::-1]

	



while(True):
	if(keyboard.is_pressed('p')):

		save_cropped_and_color_masked()

		

