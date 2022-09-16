import numpy as np
import cv2 as cv
# 马赛克1
# img1 = cv.imread('Hello.jpg')
# img2 = cv.resize(img1,(35,23))
# img3 = cv.resize(img2,(200,300))

# 马赛克2
# img1 = cv.imread('Hello.jpg')
# img2 = cv.resize(img1,(50,40))
# img3 = np.repeat(img2,10,axis=0)
# img4 = np.repeat(img3,10,axis=1)

# 马赛克3
img1 = cv.imread('Hello.jpg')
img2 = img1[::10,::10]
cv.namedWindow('img',flags=cv.WINDOW_NORMAL)
cv.resizeWindow('img',330,500)
cv.imshow('img',img2)
cv.waitKey(0)
cv.destroyAllWindows()
