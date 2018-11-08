import cv2 as cv
import numpy as np
import matplotlib as plt

def Harris(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 7)
    Ix = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=5)
    Iy = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=5)

    IxIy = np.multiply(Ix, Iy)
    Ix2  = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)

    Ix2_blur = cv.GaussianBlur(Ix2, (7, 7), 10)
    Iy2_blur = cv.GaussianBlur(Iy2, (7, 7), 10)
    IxIy_blur = cv.GaussianBlur(IxIy, (7, 7), 10)

    det = np.multiply(Ix2_blur, Iy2_blur) - np.multyply(IxIy_blur, IxIy_blur)
    trace = Ix2_blur + Iy2_blur

    R = det - 0.05 * np.multiply(trace, trace)

    plt.subplot(1, 2, 1), plt.imshow(img), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(R, cmap='gray'), plt.axis('off')


img = cv.imread("building.jpg")
Harris(img)

