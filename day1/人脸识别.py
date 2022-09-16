from email.mime import image
from tkinter import Scale
import numpy as np
import cv2 as cv

img = cv.imread('./Trla.jpg')
gray = cv.cvtColor(img,code=cv.COLOR_BGR2GRAY)
face_deactor = cv.CascadeClassifier('./haarcascade_frontalface_alt.xml')
faces = face_deactor.detectMultiScale(img,1.03,6,cv.CASCADE_SCALE_IMAGE,minSize = (100,100))
for x,y,w,h in faces:
    #cv.rectangle(gray,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)#绘制矩形,pt1、pt2为左上角和右下角,thiskness为粗细
    img2 = img[y:y+h,x:x+w]
    face = img2[::10,::10]
    face = np.repeat(face,10,axis=0)
    face = np.repeat(face,10,axis=1)
    arr = img[y:y+h,x:x+w].shape[0]
    img[y:y+h,x:x+w] = face[:h,:w]
    print(x,y,w,h)
    cv.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=[0,255,0],thickness=2)
cv.imshow('face',img)
cv.waitKey(0)
cv.destroyAllWindows()