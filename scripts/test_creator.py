# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:09:20 2018

@author: Lucio
this script create a matrix that will be used for testing using frames from 10 to 15
subject number 30 can't be analyzed. his face can't be detected
"""

import csv
import os
import face_recognition
import cv2
from PIL import Image
import numpy as np


for folder in os.listdir('.'):
    if(os.path.isdir(folder)):
        #creates the csv file for each one of the folders.
        csv_output=open('csv_files/testing/'+folder+'_testing.csv', 'wt', newline="")
        csv_writer = csv.writer(csv_output)
        path_name=folder
        count=0
        for video in os.listdir(folder):
            count += 1
            print('analizzo il video', video, count)
            video_name = video[2:5]
            video_number = int(video_name)
            vidcap = cv2.VideoCapture(folder + '/' +video)
            success, frame = vidcap.read()
            if success:
                elements_analyzed = 0
                try:
                    succ = success
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, (2*100))
                    #loop until he found 15 good faces and while video still has frames
                    while elements_analyzed < 15 and succ:
                        #reads first 10 frames to discard them
                        #because I don't want my algorithm to work on same frames 
                        #in training and testing
                        if elements_analyzed <= 9:
                            elements_analyzed += 1
                            success, frame = vidcap.read()
                        elif(elements_analyzed>9):
                            try:
                                image = Image.fromarray(frame, 'RGB')
                                #rotate the image because frames are extracted horizontally
                                image = image.rotate(-90)
                                    #uses hog features to detect the face but
                                    #could also use Convolutional Neural Net to find
                                boxes = face_recognition.face_locations(np.array(image), model="hog")
                                    #encode the features of the face
                                encode= face_recognition.face_encodings(np.array(image))
                                row = np.array(encode)
                                row = np.append(row, str(video_name))
                                if len(row) <3:
                                    print('could not find faces')
                                elif len(row) == 129:
                                    print(elements_analyzed)
                                    csv_writer.writerow(row)
                                    elements_analyzed +=1
                                else:
                                    print('found more than one face, frame discarded')                     
                            except Exception as e:
                                print('error: ', e)
                        #reads next frame
                        succ,frame=vidcap.read()                        
                    succ = False      
                except Exception as ex:
                    print('errore: ', ex)
csv_output.close()