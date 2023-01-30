# Python script that captures images from webcam video stream
# the face in photo is stored in numpy array with label as its name which user inputs
# extracts all faces from the image frame (using haarcascades)
# 

# detect largest faces and show bounding box
# then store it in numpy array

import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Face detection using haarcascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []

while True:
    ret,frame = cap.read()
    if ret==False:
        continue

    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame , 1.3 , 5)
    faces = sorted(faces , key=lambda f: f[2]*f[3])
    # print(faces)


    # pick last face because it is the largest face
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame , (x,y),(x+w,y+h) , (255,0,0),2)

        # Extract (crop out required face): region of interest
        # offset -> (frame[y,x]) so offset is padding along each side so effective box will be 10px greated
        offset = 10
        face_section = frame[y-offset:y+h+offset , x-offset:x+w+offset]
        face_section = cv2.resize(face_section , (100,100))
        # print(face_section.shape)
        # store every 10th face

        skip += 1
        if(skip%10==0):
            face_data.append(face_section)
            # print(face_data)
            print(len(face_data))

    cv2.imshow("frame",gray_frame)
    cv2.imshow("Face section",face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# convert our face list into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# np.save('data.npy',face_data)

cap.release()
cv2.destroyAllWindows()