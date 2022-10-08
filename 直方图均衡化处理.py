import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import data

moon = data.moon()

# plt.hist(moon.ravel(),bins=256)
# plt.show()

moon2 = cv.equalizeHist(moon)
plt.hist(moon2.reshape(-1),bins=256)
plt.show()

cv.imshow('moon',moon2)
cv.waitKey(0)
cv.destroyAllWindows()


