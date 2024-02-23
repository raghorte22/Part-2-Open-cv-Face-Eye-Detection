#!/usr/bin/env python
# coding: utf-8

# ### FACE & EYE DETECTION USING HAAR CASCADE CLASSIFIERS

# In[1]:


import numpy as np
import cv2

#we point opencv cascadeclassifier function to where our 
#classifier (XML file format)is stored 
#CascadeClassifier is used for object detection 

face_classifier = cv2.CascadeClassifier("D:\\Data Science with AI\\object dection\\Haarcascades\\haarcascade_frontalface_default.xml")

#load our image then convert it to grayscale
image = cv2.imread("C:\\Users\\Achal Raghorte\\OneDrive\\Pictures\\shubh 2.jpg")
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

#our classifier returns the ROI of the detected face as a tuple 
#it stores the top left coordinate and the bottom right coordinates
faces = face_classifier.detectMultiScale(gray , 1.3 ,5)

#when no faces detected , face_classifier returns and empty tuple 
if faces is ():
    print(" no face found")
    
#we iterate through our faces array and draw a rectangle 
#over each face in faces

for (x,y,w,h) in faces:
    cv2.rectangle(image ,(x,y),(x+w,y+h) ,(127,0,255) , 2)
    cv2.imshow('face detection' , image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()    


# ### LETS COMBINE FACE AND EYE DETECTION 

# In[ ]:


import numpy as np
import cv2

face_classifier =cv2.CascadeClassifier("D:\\Data Science with AI\\object dection\\Haarcascades\\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("D:\\Data Science with AI\\object dection\\Haarcascades\\haarcascade_eye.xml")

img = cv2.imread("C:\\Users\\Achal Raghorte\\OneDrive\\Pictures\\shub 3.jpg")
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)


faces = face_classifier.detectMultiScale(gray , 1.3, 5)

#when no faces detected , face_classifier returns and empty tuple 
if faces is ():
    print("No faces found")
    


for (x,y,w,h) in faces:
    cv2.rectangle(img ,(x,y) ,(x+w,y+h) ,(127,0,255) ,2)
    cv2.imshow('img' ,img)
    cv2.waitKey(0)
    roi_gray =gray[y:y+h ,x:x+w]
    roi_color =img[y:y+h , x:x+w]
    eyes= eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color ,(ex,ey) ,(ex+ew,ey+eh),(255,255,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    
cv2.destroyAllWindows()   



# In[ ]:




