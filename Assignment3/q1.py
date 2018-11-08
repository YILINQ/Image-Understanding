import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread("door.jpg")
plt.imshow(img, cmap="gray")
plt.show()

# plot four corners on the door
x1, y1 = 671.5, 320.7
x3, y3 = 769.8, 3625.68
x2, y2 = 1904.13, 112.95
x4, y4 = 1926.05, 3952.95

M, N = img.shape[0], img.shape[1]

x_1, y_1 = 100, 100
x_2, y_2 = M-1, 100
x_3, y_3 = 100, N-1
x_4, y_4 = M-1, N-1

img = cv.imread("door.jpg")

A = np.array([[x1, y1, 1, 0, 0, 0, -x_1 * x1, -x_1 * y1, -x_1],
              [0, 0, 0, x1, y1, 1, -y_1 * x1, -y_1 * y1, -y_1],

              [x2, y2, 1, 0, 0, 0, -x_2 * x2, -x_2 * y2, -x_2],
              [0, 0, 0, x2, y2, 1, -y_2 * x2, -y_2 * y2, -y_2],

              [x3, y3, 1, 0, 0, 0, -x_3 * x3, -x_3 * y3, -x_3],
              [0, 0, 0, x3, y3, 1, -y_3 * x3, -y_3 * y3, -y_3],

              [x4, y4, 1, 0, 0, 0, -x_4 * x4, -x_4 * y4, -x_4],
              [0, 0, 0, x4, y4, 1, -y_4 * x4, -y_4 * y4, -y_4]])

# use numpy method to get h

ATA = np.matmul(A.T, A)
eigenvalues, eigenvector = np.linalg.eig(ATA)
h = eigenvector[:, np.argmin(eigenvalues)]
H = h.reshape((3, 3))

img1 = cv.warpPerspective(img, H, (5000, 4000))

plt.imshow(img1), plt.show()
