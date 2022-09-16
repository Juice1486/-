
import cv2 as cv

img = cv.imread('./Trla.jpg')
r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]
cv.imshow('r',r)
cv.imshow('g',g)
cv.imshow('b',b)
cv.waitKey(0)
cv.destroyAllWindows()