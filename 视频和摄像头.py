import cv2 as cv

#视频
video = cv.VideoCapture('./2B.mp4')
#网络摄像头
# cap = cv.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# cap.set(10,100)

while True:
    success,img = video.read()
    img = cv.resize(img,(1000,800))
    cv.imshow('video',img)
    if cv.waitKey(1) & 0XFF == ord('q'):
        break
