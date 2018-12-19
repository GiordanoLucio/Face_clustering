import face_recognition
from PIL import Image, ImageDraw
import argparse



#argument parser to choose the jpg file to work on.
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--image", required=None, default='test_image.jpg',
                help="path to face to use")
ap.add_argument("-i", "--size", required=None, default=4)
args = vars(ap.parse_args())

# Load the jpg file into a numpy array
#image = face_recognition.load_image_file("foto_landmarks.jpg")
image = face_recognition.load_image_file(args["image"])
# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)
size_lines=int(args["size"])
for face_landmarks in face_landmarks_list:
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image, 'RGBA')

    # Eyebrows
    d.line(face_landmarks['left_eyebrow'], fill=(255, 255, 255, 255), width=size_lines)
    d.line(face_landmarks['right_eyebrow'], fill=(255, 255, 255, 255), width=size_lines)

	# Lips
    d.line(face_landmarks['top_lip'], fill=(255, 255, 255, 255), width=size_lines)
    d.line(face_landmarks['bottom_lip'], fill=(255, 255, 255, 255), width=size_lines)

    # Eyes
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(255, 255, 255, 255), width=size_lines)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(255, 255, 255, 255), width=size_lines)
   
    #Chin
    d.line(face_landmarks['chin'], fill=(255, 255, 255, 255), width=size_lines)
    
    #Nose
    d.line(face_landmarks['nose_bridge'], fill=(255, 255, 255, 255), width=size_lines)
    d.line(face_landmarks['nose_tip'], fill=(255, 255, 255, 255), width=size_lines)
    
    pil_image.show()