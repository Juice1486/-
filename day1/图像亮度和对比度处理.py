import cv2
import numpy as np

alpha = 0.3 # 对比度
beta = 80   # 亮度

img_path = "./Trla.jpg"
img = cv2.imread(img_path)
img2 = cv2.imread(img_path)


# 修改对比度
def updateAlpha(y):
    global alpha, img, img2
    alpha = cv2.getTrackbarPos('Alpha', 'image')#得到滑动条的值
    alpha = alpha * 0.01
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))#修改对比度，主要产生对比的原因是因为自身的初值原因

# 修改亮度
def updateBeta(y):
    global beta, img, img2
    beta = cv2.getTrackbarPos('Beta', 'image')
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))#得到亮度，数值越大，亮度就越大
    print(alpha)

# 创建窗口
cv2.namedWindow('image')
cv2.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)#创建滑动条
cv2.createTrackbar('Beta', 'image', 0, 200, updateBeta)
cv2.setTrackbarPos('Alpha', 'image', 80)#将滑动条的初始值设置为80
cv2.setTrackbarPos('Beta', 'image', 80)

while (True):#时刻刷新图像
    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()