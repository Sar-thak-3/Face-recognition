# Harcasscade classifier - already trained on many facial data
# haarcascade_frontalface_alt.xml

import cv2
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret , frame = cap.read()
    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    # faces = face_cascade.detectMultiscale(gray_frame, scalingFactor , Noofneighbours)
    # it will return the cordinates in photo where the face actually lies [(x,y,w,h)]
    faces = face_cascade.detectMultiScale(gray_frame, 1.3 , 5)

    # cv2.imshow("Video frame" , frame)
    # cv2.imshow("gray frame" , gray_frame)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,0,0),2)
    cv2.imshow("Video frame" , frame)

    key_pressed = cv2.waitKey(1) & 0xff
    if key_pressed == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# Scalefactor - parameter specifying how much the image size is reduced at each image scale
# it will shrink the image provided if scaling factor is lowr then image provided
# basically the scale factor is used to create your scale pyramid

# minNeighbours - parameter specifying how many neighbours each candidate rectangle should have to retain it
# This parameter will affect quality of detected faces. Higher the value results in less detection but with higher quality
# 3~6 is good value