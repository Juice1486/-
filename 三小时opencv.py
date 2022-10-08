import numpy as np
import cv2 as cv

img = cv.imread('./Trla.jpg')
gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
Bulr = cv.GaussianBlur(gray,(7,7),0)
cv.imshow('img',gray)
cv.imshow('img2',Bulr)
cv.waitKey(0)
cv.destroyAllWindows()