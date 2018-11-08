import cv2 as cv
import numpy as np

#  ------- Q1 -------
def correlation(I, f, mode):

    Image = I
    Filter = f
    Image_W, Image_H = Image.shape[0], Image.shape[1]
    Filter_W, Filter_H = Filter.shape[0], Filter.shape[1]
    if mode == 'valid':
        # no padding
        result = np.zeros(((Image_W - Filter_W + 1), (Image_H - Filter_H + 1), Image.shape[2]))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                for c in range(result.shape[2]):
                    result[i, j, c] = (Filter * Image[i: i + Filter_W, j: j + Filter_H, c]).sum() / Filter.sum()
        return result
    if mode == 'same':
        padded = np.zeros((Image_W + Filter_W - 1, Image_H + Filter_H - 1, Image.shape[2]))
        padded_W, padded_H = padded.shape[0], padded.shape[1]
        for c in range(padded.shape[2]):
            padded[(Filter_W - 1) // 2: padded_W - (Filter_W - 1) // 2, (Filter_H - 1) // 2: padded_H - (Filter_H - 1) // 2, c] = Image[0: Image_W, 0: Image_H, c]
        return correlation(padded, Filter, 'valid')
    if mode == 'full':
        padded = np.zeros((Image_W + 2 * Filter_W - 2, Image_H + 2 * Filter_H - 2, Image.shape[2]))
        padded_W, padded_H = padded.shape[0], padded.shape[1]
        for c in range(padded.shape[2]):
            padded[(Filter_W - 1): padded_W - (Filter_W - 1), (Filter_H - 1): padded_H - (Filter_H - 1), c] = Image[0: Image_W, 0: Image_H, c]
        return correlation(padded, Filter, 'valid')

A = cv.imread("iris.jpg")
# B = np.array([[0, 0, 0],
#               [0, 1, 0],
#               [0, 0, 0]])
# cv.imwrite("q1_valid.jpg", correlation(A, B, "valid"))
# cv.imwrite("q1_same.jpg", correlation(A, B, "same"))
# cv.imwrite("q1_full.jpg", correlation(A, B, "full"))
#  ------- Q1 -------


#  ------- Q2 -------
xx = cv.getGaussianKernel(5, 3)
yy = cv.getGaussianKernel(5, 5)
Q2 = ((xx * yy).transpose())
Q2_1 = np.flip(Q2, axis=0)
Q2_2 = np.flip(Q2_1, axis=1)
# convolution is the filter with its top to bottom, left to right, so just apply a transpose
cv.imwrite("q2.jpg", correlation(A, Q2, "same"))
#  ------- Q2 -------
