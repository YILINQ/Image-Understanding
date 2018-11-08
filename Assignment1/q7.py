import cv2 as cv
import numpy as np
img = cv.imread('portrait.jpg', 0)

#  ------- Q7 -------
blur = cv.GaussianBlur(img, (3, 3), 0)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

cv.imwrite("q7-dev-x.jpg", sobelx)
cv.imwrite("q7-dev-y.jpg", sobely)
img = cv.Laplacian(img, cv.CV_64F, ksize=3)
cv.imwrite("q7-lab.jpg", img)
#  ------- Q7 -------

