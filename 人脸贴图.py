import numpy as np
import cv2 as cv

img = cv.imread('./QQ538.jpg')
dog = cv.imread('./dog.jpg')
face_deactor = cv.CascadeClassifier('./haarcascade_frontalface_alt.xml')
faces = face_deactor.detectMultiScale(img,1.03,6,cv.CASCADE_SCALE_IMAGE,minSize = (100,100))
for x,y,w,h in faces:
    cv.rectangle(img,pt1=(x,y),pt2=(x+h,y+h),color=[0,0,255],thickness=2)
    head = cv.resize(dog,(w,h));
    img[y:y+h,x:x+w] = head
img2 = cv.resize(img,(1200,800))
cv.imshow('img',img2)
cv.waitKey(0)
cv.destroyAllWindows()