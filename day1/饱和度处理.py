from http.client import TEMPORARY_REDIRECT
import numpy as np
import cv2 as cv
import matplotlib.pylab as plt

img = cv.imread('./Trla.jpg')
img_out = img
increment = 0.5

def getHsl(x):
    global increment,img_out
    increment = cv.getTrackbarPos('Saturation','image')
    increment = (increment - 100)/100
    img_temp = img * 1.0
    img_out = img
    img_min = img_temp.min(axis = 2)
    img_max = img_temp.max(axis = 2)

    #获取空间饱和度和亮度
    delta = (img_max - img_min)/255
    value = (img_max + img_min)/255
    L = value/2
    
    mask_1 = L < 0.5
    s1 = delta/(value)
    s2 = delta/(2 - value)
    s = s1 * mask_1 + s2 * (1 - mask_1)
    if increment >= 0 :
    # alpha = increment+s > 1 ? alpha_1 : alpha_2
        temp = increment + s
        mask_2 = temp >  1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        alpha = 1/alpha -1 
        img_out[:, :, 0] = img_temp[:, :, 0] + (img_temp[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img_temp[:, :, 1] + (img_temp[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img_temp[:, :, 2] + (img_temp[:, :, 2] - L * 255.0) * alpha
        
    # 增量小于0，饱和度线性衰减
    else:
        alpha = increment
        img_out[:, :, 0] = img_temp[:, :, 0] + (img_temp[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img_temp[:, :, 1] + (img_temp[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img_temp[:, :, 2] + (img_temp[:, :, 2] - L * 255.0) * alpha
    
    # RGB颜色上下限处理(小于0取0，大于1取1)
    img_out = np.uint8(np.clip(img_out, 0, 255))

cv.namedWindow('image')
cv.createTrackbar('Saturation','image',0,200,getHsl)
cv.setTrackbarPos('Saturation','image',150)
while True:
    cv.imshow('image',img_out)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()




   
   
   
   

   
   
   
   


