import numpy as np
import cv2 as cv

img = cv.imread('R-C.jpg')
img2 = cv.resize(img,(300,400))
face = img2[42:162,138:239]
face = face[::10,::10]#每10个像素取出一个像素
face = np.repeat(face,10,axis=0)
face = np.repeat(face,10,axis=1)
print(face)
img2[42:162,138:239] = face[:120,:101]#填充保持尺寸一致
cv.imshow('波多野结衣',img2)
cv.waitKey(0)
cv.destroyAllWindows()
