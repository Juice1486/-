import numpy as np
import cv2 as cv

dog = cv.imread('./dog.jpg')
gray = cv.cvtColor(dog,cv.COLOR_RGB2GRAY)
gray2 = cv.GaussianBlur(gray,(5,5),0)#高斯平滑
canny = cv.Canny(gray2,75,200)
cv.imshow('canny',canny)
# counters,layers = cv.findContours(gray,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# img = cv.drawContours(gray,counters,3,(0,255,0),3)
cv.waitKey(0)
cv.destroyAllWindows()