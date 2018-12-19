# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:07:22 2018

@author: Lucio

#This is the first script I wrote for feature extraction. 
It was executed after extracting the frames from the video, in fact, it works taking
the frames in input.
I can't upload the dataset for privacy problem but as you can see using my face
it extracts the distances and normalize them dividing every distance calculated by 
the length of the diagonal of the boundary box.
The script returns the matrix with the distances. One entry row per face/frame
This script extracted the faces from all the frames of the video. 
Later this became a problem, in fact, the clusters were not balanced, and to solve this problem I changed 
the algorithm to only extract 10 good faces per video in training and 5, different from the ones used for 
the training, for testing.
The way in which I avoid getting more than 10 entries per video, can be seen in the other script. I prefered 
to put this older version to make you understand how was the first script and how it has evolved.

"""

from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import imutils
import dlib
import cv2
import math
import csv
import os
import re

detector = dlib.get_frontal_face_detector()
#shape predictor used to detect the 68 points
predictor = dlib.shape_predictor('../different_feature_extractor/shape_predictor_68_face_landmarks.dat')
root = '../dataset/frames/7_3D'
#this is the path were the file.csv is saved
csv_output= open("../csv_files/68_points_results/feature_extracted_68_points.csv", 'wt', newline="") #creazione csv #@@@@@@@@ CAMBIARE QUI @@@@@@@@@@@@@@
csv_writer=csv.writer(csv_output)		   	   #dove scrive il csv
for folder_name in os.listdir(root):
    print('analizzo la cartella', folder_name)
    numero_persona=re.sub("[^0-9]", "", folder_name[2:])
        #IMPORTANT: GOTTA GIVE A DYNAMIC WAY OF NAMING
    for filename in os.listdir(root+'/'+folder_name): #@@@@@@@@ CAMBIARE QUI @@@@@@@@@@@@@@
      if filename.endswith('.jpg'):
        print(filename)
        image=None
        rect=None
        rects=None
        x=None
        y=None
        w=None
        h=None
        shape=None
       # load the input image, resize it, and convert it to grayscale
        try: 
          image = cv2.imread(root+'/'+folder_name+'/'+filename)
          image = imutils.resize(image, width=500)
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          rects = detector(gray, 1)
          # loop over the face detections
          for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then 
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #loop over the (x, y)-coordinates for the facial landmarks # and draw them on the image
            for (x, y) in shape:
              cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            arrxi=[] #array for i points
            arryi=[] #array for j points
            for indice in shape:
              x1 = str(indice[0])
              y1 = str(indice[1])
              arrxi.append(x1) #add x1 all'array di i
              arryi.append(y1) #add y1 all'array di j
          count=0
          i=0
          distanza=0
          date=[]
          while i < len(arrxi):
            j=i
            while j < len(arrxi):
              if i==j:
                j+=1
                continue
              #calculates the Eucledian distance i-j
              distanza=round(math.sqrt(math.pow((int(arrxi[j])-int(arrxi[i])), 2)+math.pow((int(arryi[j])-int(arryi[i])),2)),4)
              #Normalizing: divides the distance calculated before by the diagonal of the boundary box
              distanza=distanza/ math.sqrt(math.pow(h,2) + math.pow(w,2)) 
              distanza = "{:.7f}".format(distanza)
              date.append(str(distanza))
              j+=1
              count+=1
            i+=1
          date.append(numero_persona)
          csv_writer.writerow(date)
          #print( 'w:',w, ', h: ', h, ' ,', date)
          #print('file analizzato con successo ' + filename + count) #usata per controllare quanti elementi scriveva nel csv => 68*68.
        except: 
          print('file non utilizzabile ', filename)
        #  os.remove(filename) #if you want to remove the file after it's features have been saved on the csv
csv_output.close()