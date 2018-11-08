import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

def find_K_2d_to_3d():
    img = cv.imread("q3.jpg")
    plt.imshow(img), plt.show()
    M, N = img.shape[0], img.shape[1]
    x1, y1, X1, Y1, Z1 = 79.75, 281.57, -150 / M, 120 / N, 500
    x2, y2, X2, Y2, Z2 = 114.8, 565.9, -150 / M, 0, 500
    x3, y3, X3, Y3, Z3 = 149.9, 854.3, -150 / M, -120 / N, 500
    x4, y4, X4, Y4, Z4 = 527.8, 262.1, 0, 120 / N, 500
    x5, y5, X5, Y5, Z5 = 531.7, 581.5, 0, 0, 500
    x6, y6, X6, Y6, Z6 = 539.5, 854.3, 0, -120 / N, 500

    A = np.array([[X1, Y1, Z1, 1, 0, 0, 0, 0, -x1*X1, -x1*Y1, -x1*Z1, -x1],
                  [0, 0, 0, 0, X1, Y1, Z1, 1, -y1*X1, -y1*Y1, -y1*Z1, -y1],

                  [X2, Y2, Z2, 1, 0, 0, 0, 0, -x2 * X2, -x2 * Y2, -x2 * Z2, -x2],
                  [0, 0, 0, 0, X2, Y2, Z2, 1, -y2 * X2, -y2 * Y2, -y2 * Z2, -y2],

                  [X3, Y3, Z3, 1, 0, 0, 0, 0, -x3 * X3, -x3 * Y3, -x3 * Z3, -x3],
                  [0, 0, 0, 0, X3, Y3, Z3, 1, -y3 * X3, -y3 * Y3, -y3 * Z3, -y3],

                  [X4, Y4, Z4, 1, 0, 0, 0, 0, -x4 * X4, -x4 * Y4, -x4 * Z4, -x4],
                  [0, 0, 0, 0, X4, Y4, Z4, 1, -y4 * X4, -y4 * Y4, -y4 * Z4, -y4],

                  [X5, Y5, Z5, 1, 0, 0, 0, 0, -x5 * X5, -x5 * Y5, -x5 * Z5, -x5],
                  [0, 0, 0, 0, X5, Y5, Z5, 1, -y5 * X5, -y5 * Y5, -y5 * Z5, -y5],

                  [X6, Y6, Z6, 1, 0, 0, 0, 0, -x6 * X6, -x6 * Y6, -x6 * Z6, -x6],
                  [0, 0, 0, 0, X6, Y6, Z6, 1, -y6 * X6, -y6 * Y6, -y6 * Z6, -y6]])

    ATA = np.matmul(A.transpose(), A)
    v = np.linalg.svd(A)[2].T[-1].reshape((3, 4))
    v = v[0:3, 0:3]
    result = np.linalg.qr(v)[1]
    print(result)

# find_K_2d_to_3d()

def find_K_straight():
    d = 50 * 10

    img = cv.imread("q3.jpg")
    plt.imshow(img), plt.show()

    M, N = img.shape[0], img.shape[1]
    center_x, center_y = M / 2, N / 2
    pixel_size = 25.4 / 227    # ppi
    print(center_x, center_y)

    length_wise = 180
    pixel_wise = (center_y - 265)

    # find f
    f = d * (pixel_wise / length_wise) * pixel_size
    # find px and py
    ox, oy = center_y, center_x

    K_intrinsic = np.array([[f, 0, ox],
                            [0, f, oy],
                            [0, 0, 1]])
    print(K_intrinsic)

# find_K_straight()
def find_K_vanishing_point():
    img = cv.imread("vanishingPoint.jpg")
    plt.imshow(img), plt.show()
    originx, originy = 755.794, 759.24
    x1, y1, z1 = 79.364 - originx, 962.338 - originy, 1
    x2, y2, z2 = 1253.1 - originx, 921.516 - originy, 1
    x3, y3, z3 = 1079.49 - originx, 1986.65 - originy, 1

    # x1, y1, z1 = 212, 2138, 1
    # x2, y2, z2 = -49, 42, 1
    # x3, y3, z3 = 1105, 146, 1

    A = np.array([[x1*x2 + y1*y2, x2*z1 + x1*z2, z1*y2 + y1*z2, z1*z2],
                  [x1*x3 + y1*y3, x3*z1 + x1*z3, z3*y1 + y3*z1, z1*z3],
                  [x2*x3 + y2*y3, x3*z2 + x2*z3, z2*y3 + z3*y2, z2*z3]])
    # print(A)
    ATA = np.matmul(A.T, A)
    #print(ATA)
    # find null space of A
    eigenvalues, eigenvector = np.linalg.eig(ATA)
    h = eigenvector[:, np.argmin(eigenvalues)]
    # print(h)
    h = np.linalg.svd(A)[2][-1]
    #print(h)
    w1 = h[0]
    w2 = h[1]
    w3 = h[2]
    w4 = h[3]

    #print(w1, w2, w3, w4)
    w = np.array([[w1, 0, w2],
                  [0, w1, w3],
                  [w2, w3, w4]])
    # print(w)
    # K = inv(chol(W))
    K = np.linalg.inv(np.linalg.cholesky(w).T)

    print(K / K[2, 2])
find_K_vanishing_point()
