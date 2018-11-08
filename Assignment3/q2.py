import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import skimage as sk
from skimage.transform import AffineTransform, ProjectiveTransform
from skimage.measure import ransac


def find_match(img, threshold):
    # open source implementation and modification from following link:
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    MIN_MATCH_COUNT = 200
    img2 = cv.imread(img)          # queryImage
    img1 = cv.imread('bookCover.jpg') # trainImage
    outimage = img
    # img1 = cv2.imread('img2.jpg', 0) # trainImage
    # img1 = cv2.imread('img3.jpg',0) # trainImage
    # Initiate SIFT detector
    sift = cv.xfeatures2d_SIFT.create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append(m)
    # import random
    # random.shuffle(good)
    # good = good[:20]
    print(len(good))
    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv.imwrite(outimage + "---.jpg", img3)
    return kp1, kp2, good

def Q2_b_affine(P, p):
    return (np.log(1 - P)) / (np.log(1 - pow(p, 3)))

def Q2_b_homography(P, p):
    return (np.log(1 - P)) / (np.log(1 - pow(p, 4)))

# part c-d-e is from this link:
# http://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html
def ransac_transformation(mode, img1, img2, outimage):
    kp1, kp2, good = find_match(img2, 0.55)

    img2 = cv.imread(img2)          # queryImage
    img1 = cv.imread(img1)                      # trainImage

    src = np.array([kp1[m.queryIdx].pt for m in good])
    dst = np.array([kp2[m.trainIdx].pt for m in good])
    if mode == "affine":
        model_robust, inliers = sk.measure.ransac((src, dst), AffineTransform, min_samples=5, residual_threshold=2,
                                              max_trials=100)
        h = model_robust.params
        np.delete(h, 2, axis=0)
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                src = np.array([[j],
                                [i],
                                [1]])
                dst = np.matmul(h, src)

                y = int(dst[0, 0])
                x = int(dst[1, 0])

                img2[x, y] = img1[i, j]
        cv.imwrite(outimage, img2)

    elif mode == "homography":
        model_robust, inliers = sk.measure.ransac((src, dst), ProjectiveTransform, min_samples=5, residual_threshold=2,
                                                  max_trials=1000)
        h = model_robust.params
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                src = np.array([[j],
                                [i],
                                [1]])
                dst = np.matmul(h, src)

                y = int((dst[0, 0] / dst[2, 0]))
                x = int((dst[1, 0] / dst[2, 0]))

                img2[x, y] = img1[i, j]
        cv.imwrite(outimage, img2)
    else:
        return False

# find_match(0.85)
if __name__ == "__main__":
    #pass
    # find_match("img1.jpg", 0.95)
    # find_match("img2.jpg", 0.95)
    # find_match("img3.jpg", 0.95)
    # print("Iterations needed for affine")
    # print(Q2_b_affine(0.99, 0.90))
    # print(Q2_b_affine(0.99, 0.90))
    # print(Q2_b_affine(0.99, 0.80))
    #
    # print("-------------------------------")
    #
    # print("Iterations needed for homography")
    # print(Q2_b_homography(0.99, 0.90))
    # print(Q2_b_homography(0.99, 0.90))
    # print(Q2_b_homography(0.99, 0.80))
    # find_match("img3.jpg", 0.95)
    # ransac_transformation("affine", 'bookCover.jpg', "img1.jpg", "q2_c_1.jpg")
    # ransac_transformation("affine", 'bookCover.jpg', "img2.jpg", "q2_c_2.jpg")
    # ransac_transformation("affine", 'bookCover.jpg', "img3.jpg", "q2_c_3.jpg")
    # ransac_transformation("homography", 'bookCover.jpg', "img1.jpg", "q2_d_1.jpg")
    # ransac_transformation("homography", 'bookCover.jpg', "img2.jpg", "q2_d_2.jpg")
    # ransac_transformation("homography", 'bookCover.jpg', "img3.jpg", "q2_d_3.jpg")
    #
    ransac_transformation("homography", 'anotherBookCoverRed.jpg', "anna1.jpg", "q2_e_1_1.jpg")
    ransac_transformation("homography", 'anotherBookCoverRed.jpg', "anna2.jpg", "q2_e_2_2.jpg")
    ransac_transformation("homography", 'anotherBookCoverRed.jpg', "anna3.jpg", "q2_e_3_3.jpg")
