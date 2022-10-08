import numpy as np
import cv2 as cv

img = cv.imread('./Trla.jpg')
img2 = img[:, :, 0]
print((img[0,1]>200))
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()





