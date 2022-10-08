import cv2
import numpy as np

def Gaussnoise_func(image, mean=0.1, var=0.005):
    image = np.array(image/255, dtype=float)                   
    noise = np.random.normal(mean, var ** 0.5, image.shape)    
    out = image + noise                                        
    if out.min() < 0:
     low_clip = -1.
    else:
     low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    cv2.imshow('gaosi',out)
    cv2.imwrite('test1.jpg',out)

def salt_noise(peppers):
    row, column ,c= peppers.shape
    noise_salt = np.random.randint(0, 256, (row, column ,3))
    noise_pepper = np.random.randint(0, 256, (row, column ,3))
    rand = 0.05
    noise_salt = np.where(noise_salt < rand * 256, 255, 0)
    noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
    peppers.astype("float")
    noise_salt.astype("float")
    noise_pepper.astype("float")
    salt = peppers + noise_salt
    pepper = peppers + noise_pepper
    salt = np.where(salt > 255, 255, salt)
    pepper = np.where(pepper < 0, 0, pepper)
    cv2.imshow("salt", salt.astype("uint8"))
    cv2.imshow("pepper", pepper.astype("uint8"))
    cv2.imwrite('test2.jpg',salt.astype('uint8'))
    cv2.imwrite('test3.jpg',pepper.astype('uint8'))

def MedianFilter(img,K_size=3):
    # 中值滤波 
    h,w,c = img.shape
    # 零填充
    pad = K_size//2
    out = np.zeros((h + 2*pad,w + 2*pad,c),dtype=float)
    out[pad:pad+h,pad:pad+w] = img.copy().astype(float)
    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad+y,pad+x,ci] = np.median(tmp[y:y+K_size,x:x+K_size,ci])
    out = out[pad:pad+h,pad:pad+w].astype(np.uint8)
    cv2.imshow('MedianFIlter',out)

def mean_filter(image):
    K = ([1, 1, 1],
         [1, 1, 1],
         [1, 1, 1])
    K = np.array(K)
    H, W, C = image.shape
    result = image.copy()
    # 因为卷积核是以左上角为定位，所以遍历时最后要停到H-2处
    for h in range(1, H-2):
        for w in range(1, W-2):
            for c in range(C):
                result[h, w, c] = sum(sum(K * result[h:h+K.shape[0], w:w+K.shape[1], c])) // 9
    cv2.imshow('mean_filter',result)
    return result

def Gaussian_Filter(img):
    h,w,c = img.shape
    # 高斯滤波 
    K_size = 3
    sigma = 1.3
    
    # 零填充
    pad = K_size//2
    out = np.zeros((h + 2*pad,w + 2*pad,c),dtype=float)
    out[pad:pad+h,pad:pad+w] = img.copy().astype(float)
    
    # 定义滤波核
    K = np.zeros((K_size,K_size),dtype=float)
    
    for x in range(-pad,-pad+K_size):
        for y in range(-pad,-pad+K_size):
            K[y+pad,x+pad] = np.exp(-(x**2+y**2)/(2*(sigma**2)))
    K /= (sigma*np.sqrt(2*np.pi))
    K /=  K.sum()
    
    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad+y,pad+x,ci] = np.sum(K*tmp[y:y+K_size,x:x+K_size,ci])
    
    out = out[pad:pad+h,pad:pad+w].astype(np.uint8)
    cv2.imshow('gaosi_filter.jpg',out)
    return out

#peppers = cv2.imread("R-C.jpg")
salt_img = cv2.imread('test2.jpg')
gaosi_img = cv2.imread('test1.jpg')
salt_black_img = cv2.imread('test3.jpg')
cv2.imshow('gaosinoise',gaosi_img)
# salt_noise(peppers)
# Gaussnoise_func(peppers)
#MedianFilter(salt_img)
Gaussian_Filter(gaosi_img)
#mean_filter(salt_img)



















cv2.waitKey()


