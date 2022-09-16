import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

th = 0.5
img_path = './Trla.jpg'
img = cv.imread(img_path)
img2 = np.copy(img)

def RGB2GRAY(img):
    return 0.21*img[:,:,0]+0.71*img[:,:,1]+0.07*img[:,:,2]

def Binarzation(x):
    global th,img2
    gray = RGB2GRAY(img/255)
    bin = np.copy(gray)
    th = cv.getTrackbarPos('2-value', 'image')#得到滑动条的值
    th = th * 0.01
    bin[gray >= th] = 1
    bin[gray < th] =0
    img2 = bin

gray = RGB2GRAY(img/255)
cv.namedWindow('image')
cv.createTrackbar('2-value','image',0,100,Binarzation)
cv.setTrackbarPos('2-value','image',50)
while (True):#时刻刷新图像
    cv.imshow('image',img2)
    if cv.waitKey(1) == ord('q'):
        break
cv.waitKey(0)
cv.destroyAllWindows()

# plt.imshow(img)
# plt.figure()
# plt.imshow(gray,cmap='gray')
# plt.show()
