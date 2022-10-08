from typing import no_type_check_decorator
import numpy as np
import cv2 as cv


def salt_noise(img):
    h, w, c = img.shape
    # 生成椒盐噪声，一个为黑色，一个为白色
    noise_salt = np.random.randint(0, 256, (h, w, c))
    noise_pepper = np.random.randint(0, 256, (h, w, c))
    rand = 0.05
    # 将图片中小于这个值的点改为全黑或全白
    noise_salt = np.where(noise_salt < rand*256, 255, 0)
    noise_pepper = np.where(noise_pepper < rand*256, -255, 0)
    # 类型转换
    img.astype('float')
    noise_pepper.astype('float')
    noise_salt.astype('float')
    # 给图片增加噪声
    salt = img+noise_salt
    pepper = img+noise_pepper
    # 限定范围
    salt = np.uint8(np.clip((salt), 0, 255))
    pepper = np.uint8(np.clip((pepper), 0, 255))
    cv.imshow('white', salt)
    cv.imshow('black', pepper)


def Gaussnoise_func(img, mean=0.1, var=0.005):
    # 图像归一化处理
    img = np.array(img/255, dtype=float)
    h, w, c = img.shape
    # 生成高斯噪声
    noise = np.random.normal(mean, var**0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low = -1.
    else:
        low = 0.
    # 限定范围
    out = np.clip(out, low, 1.0)
    out = np.uint8(out*255)
    cv.imshow('test', out)


def Medium_Filter(img, K_size=3):
    h, w, c = img.shape

    pad = K_size//2
    out = np.zeros((h+2*pad, w+2*pad, c), dtype=float)
    out[pad:h+pad, pad:w+pad] = img.copy().astype(float)
    tmp = out.copy()

    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad+y, pad+x,
                    ci] = np.median(tmp[y:y+K_size, x:x+K_size, ci])
    out = out[pad:h+pad, pad:w+pad]
    out = np.uint8(out)
    cv.imshow('medium_filter', out)


def mean_filter(img, K_size=3):
    h, w, c = img.shape
    K = np.ones((3, 3))
    result = img.copy()
    for y in range(1, h-2):
        for x in range(1, w-2):
            for ci in range(c):
                result[y, x, ci] = sum(
                    sum(K*result[y:y+K_size, x:x+K_size, ci]))//9
    result = np.clip(np.uint8(result), 0, 255)
    cv.imshow('mean_filter', result)


def Gaussian_Filter(img, sigma=1.5, K_size=3):
    h,w,c = img.shape

    pad = K_size//2
    out = np.zeros((2*pad+h,2*pad+w,c),dtype=float)
    out[pad:pad+h,pad:pad+w] = img.copy().astype(float)

    K=np.zeros((K_size,K_size))
    tmp = out.copy()
    for y in range(-pad,-pad+K_size):
        for x in range(-pad,-pad+K_size):
            K[y+pad,x+pad] =  np.exp(-(x**2+y**2)/(2*(sigma**2)))
    K /= (sigma*np.sqrt(2*np.pi))
    K /=  K.sum()

    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[y+pad,x+pad,ci] = np.sum(K*tmp[y:y+K_size,x:x+K_size,ci])
    out = np.clip(np.uint8(out),0,255)
    cv.imshow('Gaussian_Filter',out)


img = cv.imread('R-C.jpg')
img1 = cv.imread('test2.jpg')
img2 = cv.imread('test1.jpg')
img3 = cv.imread('test3.jpg')
# salt_noise(img)
# Gaussnoise_func(img)
# Medium_Filter(img1)
# Medium_Filter(img3)
#mean_filter(img1)
#mean_filter(img3)
Gaussian_Filter(img2)
cv.waitKey(0)
cv.destroyAllWindows()
