# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:09:20 2018

@author: Lucio
This script tries to extract 10 features from every video of every folder
on subject 33 of folder 7_3D the algorithm can't find any face.
"""

import csv
import os
import face_recognition
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#creates the matrix used for training
csv_output=open('../csv_files/training_matrix.csv', 'wt', newline="")
csv_writer = csv.writer(csv_output)
print('starting')
path='../dataset/folders/'
for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        print('working on folder: ', folder)
        for video in os.listdir(path + '/' +folder):
            print('analyzing video: ', video)
            video_name = video[2:5]
            vidcap = cv2.VideoCapture(path + folder + '/' +video)
			#load a frame in the variable 'frame' 
            success, frame = vidcap.read()
            #if the frame has been read succesfully
            if success:
				#set number of elements analyzed = 0
                elements_analyzed = 0
                try:
                    succ = success
					#start from 0.2s 
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, (2*100))
						#encodes the first 10 good faces found in the video.
                    while elements_analyzed < 10 and succ:
                        try:
                            #data=[]
                            image = Image.fromarray(frame, 'RGB')
                            image = image.rotate(-90)
							#uses hog feature but could also use 'cnn' as parameter to use Convolutional Neural Net to find the faces.
                            boxes = face_recognition.face_locations(np.array(image), model="hog")
                            encode= face_recognition.face_encodings(np.array(image))
                            row = np.array(encode)
                            row = np.append(row, str(video_name))
                            if len(row) <3:
                                print('could not find a face')
								#the algorithm could find more then one face, so one would be a false positive.
								#to avoid this, I discard every frame where it finds more faces.
								#Remember: every face is encoded with one 128-d vector.
                            elif len(row) == 129:
                                csv_writer.writerow(row)
                                elements_analyzed +=1
                                print(elements_analyzed)
                            else:
                                print('Frame discarded: found more than one face')                     
                        except Exception as e:
                            print('errore: ', e)
							#reads the next frame
                        succ,frame=vidcap.read()
                    succ = False      
                except Exception as ex:
                        print('errore: ', ex)
csv_output.close()
