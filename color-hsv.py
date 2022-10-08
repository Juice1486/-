from email.mime import image
import cv2 as cv
import numpy as np
#读取图片
img1 = cv.imread('Hello.jpg')

img2=cv.cvtColor(img1,code = cv.COLOR_RGB2HSV)

#定义HSV颜色空间中蓝色的范围
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
#根据蓝色的范围标记图片中那些位置是我们的蓝色
mask = cv.inRange(img2,lower_blue,upper_blue)
res = cv.bitwise_and(img1,img1,mask = mask)
#print(mask)
print(res)
#显示图片
cv.imshow('img',res)
#延时
cv.waitKey(0)
cv.destroyAllWindows()

print('Hello')

