# ------ Q11 ------
import cv2 as cv
img = cv.imread('portrait.jpg')
edges = cv.Canny(img, 180, 485)

cv.imwrite("q11.jpg", edges)
# ------ Q11 ------
