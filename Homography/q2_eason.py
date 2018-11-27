import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import skimage as sk
from skimage.transform import AffineTransform
from skimage.measure import ransac

def SIFT_feature_matching():

    img1 = cv.imread("findBook.png")
    img2 = cv.imread("book.jpeg")

    sift = cv.xfeatures2d.SIFT_create()
    key1, desc1 = sift.detectAndCompute(img1, None)
    key2, desc2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    print(good)

    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, flags = 2)

    img3 = cv.drawMatches(img1,key1,img2,key2,good,None,**draw_params)

    plt.imshow(img3), plt.show()

    return good, key1, key2


def RANSAC_affine(good, key1, key2):

    img1 = cv.imread("findBook.png")
    img2 = cv.imread("book.jpeg")

    src = np.array([key1[m.queryIdx].pt for m in good])
    dest = np.array([key2[m.trainIdx].pt for m in good])

    print(src)
    print(dest)
    model_robust, inliers = sk.measure.ransac((src, dest), AffineTransform, min_samples=10,
                                              residual_threshold=2, max_trials=1000)

    aff = model_robust.params
   # aff = np.delete(aff, 2, axis=0)
    img1 = cv.warpPerspective(img1, aff, (3900, 3500))

    print(aff)

    for row in range(img1.shape[0]):
        for col in range(img1.shape[1]):

            src = np.array([[col], [row], [1]])

            dest = np.matmul(aff, src)

            y = int(np.round(dest[0,0]))
            x = int(np.round(dest[1,0]))

            img2[x, y] = img1[row, col]



    return

def RANSAC_hompgraphy(good, key1, key2):

    img1 = cv.imread("findBook.png")
    img2 = cv.imread("book.jpeg")

    src = np.array([key1[m.queryIdx].pt for m in good])
    dest = np.array([key2[m.trainIdx].pt for m in good])

    src = (src.reshape(-1,1,2)).astype(np.float)
    dest = (dest.reshape(-1, 1, 2)).astype(np.float)

    M, mask = cv.findHomography(src, dest, cv.RANSAC, 5.0)

    print(M)

    matchesMask = mask.ravel().tolist()

    #--------------------
    h, w = img1.shape[0], img1.shape[1]
    pts = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    pts = pts.astype(np.float)
    dst = cv.perspectiveTransform(pts, M)

    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv.drawMatches(img1, key1, img2, key2, good, None, **draw_params)

    plt.imshow(img3, 'gray'), plt.show()

    for row in range(img1.shape[0]):
        for col in range(img1.shape[1]):

            src = np.array([[col], [row], [1]])

            dest = np.matmul(M, src)

            y = int(np.round(dest[0,0]/dest[2,0]))
            x = int(np.round(dest[1,0]/dest[2,0]))

            img2[x, y] = img1[row, col]



a, b, c = SIFT_feature_matching()

RANSAC_affine(a, b, c)
RANSAC_hompgraphy(a, b, c)
