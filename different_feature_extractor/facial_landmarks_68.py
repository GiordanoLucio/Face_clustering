#per ogni immagine in input, questo programma da in output un file .txt
#contenente:
# nome dell'immagine
# coordinate del quadrato che contiene il volto (punto x in alto a sinistra, punto y in alto a sinistra, altezza e larghezza)
# 68 coordinate x y ordinate di tutti i landmark trovati

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
import csv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=None, default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=None, default="test_image.jpg",
                help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
percorso = "file1.txt"
myfile = open(percorso, 'w')


# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    
    
    #modifica PAPS
    line1 = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
    myfile.write(args["image"] + "\n")
    myfile.write(line1)
    

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    arrxi=[] #array per i punti i
    arryi=[] #array per i punti j
    for indice in shape:
        x1 = str(indice[0])
        y1 = str(indice[1])
        myfile.write("("+x1 + ", " + y1 + ") ")
        arrxi.append(x1) #aggiungo x1 all'array di i
        arryi.append(y1) #aggiungo y1 all'array di j

    myfile.write("\n")

myfile.close()

percorso = args["image"]+(".txt") #il nome del volto su cui ho eseguito lo script
count=0
i=0
distanza=0

pathfile=args["image"]
pat = pathfile[:-4]

csv_output= open(pat+".csv", 'wt', newline="") #creazione csv
csv_writer=csv.writer(csv_output)		   	   #dove scrive il csv

while i < len(arrxi):#ok, fa da 0 a 67, quindi 68 volte
	j=0
	date=[]
	while j < len(arrxi):
		#if i==j:#j+=1#continue #utilizzato per non far computare quando j=i ovvero non calcola la distanza fra un punto e se stesso(che sarebbe 0)
		distanza=round(math.sqrt(math.pow((int(arrxi[j])-int(arrxi[i])), 2)+math.pow((int(arryi[j])-int(arryi[i])),2)),2)
		date.append(str(distanza))
##		risultato.write(str(distanza) + " ")
		j+=1
		count+=1
##	risultato.write("\n")
	#print(date)
	csv_writer.writerow(date)	#scrive la row nel csv
	i+=1

csv_output.close()
print('distanze calcolate: ',count) #elementi della matrice
##risultato.close()	#chiusura file risultato.txt

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
