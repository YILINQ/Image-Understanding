import numpy as np
import imutils
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()


    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False, offset=0):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)
        return (matches, H, status)

def stich(img1, img2, offset=0, flip=np.fliplr, order=False):
    imageA = cv2.imread(img1)
    imageB = cv2.imread(img2)
    outimage = img1 + img2
    imageA = imutils.resize(imageA, width=400)
    imageB = imutils.resize(imageB, width=400)
    if order:
        imageA = np.fliplr(imageA)
        imageB = np.fliplr(imageB)
    # stitch the images together to create a panorama
    stitcher = Stitcher()
    (result) = stitcher.stitch([imageA, imageB], showMatches=False)
    if flip:
        result = flip(result)
    cv2.imwrite(outimage, result)
    return outimage

if __name__ == "__main__":
    # # image_4_5 = stich("landscape_4.jpg", "landscape_5.jpg")
    # # image_5_6 = stich("landscape_5.jpg", "landscape_6.jpg")
    # # # image_6_7 = stich("landscape_6.jpg", "landscape_7.jpg")
    # # # image_3_4 = stich("landscape_3.jpg", "landscape_4.jpg")
    # # image_4_5_6 = stich(image_4_5, image_5_6)
    # # img_4_5 = cv2.imread(image_4_5)
    # # # plt.imshow(img_4_5), plt.show()
    # # y_cutoff_4_5 = img_4_5.shape[0]
    # # # cutoff x=117
    # # new_img_4_5 = np.zeros((533, img_4_5.shape[1] - 125, 3))
    # # new_img_4_5[:, :, :] = img_4_5[:, 125:, :]
    # # new_img_4_5 = np.fliplr(new_img_4_5)
    # # # plt.imshow(new_img_4_5), plt.show()
    # # cv2.imwrite("4-5.jpg", new_img_4_5)
    # # order
    #
    # image_5_6 = stich("landscape_5.jpg", "landscape_6.jpg", flip=None)
    # img_5_6 = cv2.imread(image_5_6)
    # # plt.imshow(img_5_6), plt.show()
    # # cutoff2 = 570
    # new_img_5_6 = np.zeros((533, 630, 3))
    # new_img_5_6[:, :, :] = img_5_6[:, :630, :]
    # #plt.imshow(new_img_4_5_6), plt.show()
    # cv2.imwrite("56.jpg", new_img_5_6)
    #
    # image_5_6_7 = stich("56.jpg", "landscape_7.jpg", flip=None)
    # img_5_6_7 = cv2.imread(image_5_6_7)
    # #plt.imshow(img_4_5_6_7), plt.show()
    #
    # # cutoff = 560
    # new_img_5_6_7 = np.zeros((533, 558, 3))
    # new_img_5_6_7[:, :, :] = img_5_6_7[:, :558, :]
    # # plt.imshow(new_img_4_5_6), plt.show()
    # cv2.imwrite("567.jpg", new_img_5_6_7)
    #
    # # image_8_9 = stich("landscape_8.jpg", "landscape_9.jpg", flip=None)
    # image_5_6_7_8 = stich("567.jpg", "landscape_8.jpg", flip=None)
    # img_5_6_7_8 = cv2.imread(image_5_6_7_8)
    # # plt.imshow(img_5_6_7_8), plt.show()
    # # cutoff = 591
    # new_img_5_6_7_8 = np.zeros((533, 588, 3))
    # new_img_5_6_7_8[:, :, :] = img_5_6_7_8[:, :588, :]
    # #plt.imshow(new_img_4_5_6_7_8), plt.show()
    # cv2.imwrite("5678.jpg", new_img_5_6_7_8)
    #
    # image_5_6_7_8_9 = stich("5678.jpg", "landscape_9.jpg", flip=None)
    # img_5_6_7_8_9 = cv2.imread(image_5_6_7_8_9)
    # #plt.imshow(img_4_5_6_7_8_9), plt.show()
    # # no need to remove
    # cv2.imwrite("56789.jpg", img_5_6_7_8_9)
    #
    #
    #
    # image_5_4 = stich("landscape_5.jpg", "landscape_4.jpg", flip=np.fliplr, order=True)
    # img_5_4 = cv2.imread(image_5_4)
    # #plt.imshow(img_5_4), plt.show()
    # # cutoff = 128
    # y_cutoff_34 = img_5_4.shape[1]
    # new_img_5_4 = np.zeros((533, y_cutoff_34 - 130, 3))
    # new_img_5_4[:, :, :] = img_5_4[:, 130:, :]
    # #plt.imshow(new_img_5_4), plt.show()
    #
    # cv2.imwrite("54.jpg", new_img_5_4)
    # # reverse order to stich 1-4
    # image_3_4 = stich("54.jpg", "landscape_3.jpg", flip=np.fliplr, order=True)
    # img_3_4 = cv2.imread(image_3_4)
    # #plt.imshow(img_3_4), plt.show()
    # # cutoff = 239
    # y_cutoff_34 = img_3_4.shape[1]
    # new_img_3_4 = np.zeros((533, y_cutoff_34 - 249, 3))
    # new_img_3_4[:, :, :] = img_3_4[:, 249:, :]
    # # plt.imshow(new_img_3_4), plt.show()
    # cv2.imwrite("345.jpg", new_img_3_4)
    #
    # # imgA = cv2.imread("34.jpg")
    # # imgB = cv2.imread("456789.jpg")
    # # cv2.imwrite("456789.jpg", imgB)
    # # image___ = stich("34.jpg", "456789.jpg", flip=None, order=False)
    # image_2_3_4 = stich("345.jpg", "landscape_2.jpg", flip=np.fliplr, order=True)
    # img_2_3_4 = cv2.imread(image_2_3_4)
    # #plt.imshow(img_2_3_4), plt.show()
    # #cutoff = 238
    # y_cutoff_234 = img_2_3_4.shape[1]
    # new_img_2_3_4 = np.zeros((533, y_cutoff_34 - 261, 3))
    # new_img_2_3_4[:, :, :] = img_2_3_4[:, 261:, :]
    # #plt.imshow(new_img_2_3_4), plt.show()
    # cv2.imwrite("2345.jpg", new_img_2_3_4)
    #
    # image_1_2_3_4 = stich("2345.jpg", "landscape_1.jpg", flip=np.fliplr, order=True)
    # img_1_2_3_4 = cv2.imread(image_1_2_3_4)
    # #plt.imshow(img_1_2_3_4), plt.show()
    # # cutoff = 265
    # y_cutoff_1234 = img_1_2_3_4.shape[1]
    # new_img_1_2_3_4 = np.zeros((533, y_cutoff_34 - 275, 3))
    # new_img_1_2_3_4[:, :, :] = img_1_2_3_4[:, 275:, :]
    # #plt.imshow(new_img_1_2_3_4), plt.show()
    # cv2.imwrite("12345.jpg", new_img_1_2_3_4)
    #
    #
    #
    # # image___ = stich("12345.jpg", "56789.jpg", flip=None, order=False)
    # imga = cv2.imread("12345.jpg")
    # imgb = cv2.imread("56789.jpg")
    # # plt.imshow(imga), plt.show()
    # # plt.imshow(imgb), plt.show()
    #
    # # cv2.imwrite("123456789.jpg", img)
    # # 407 ---> cut
    # # 12  <--- cut
    # img_left = np.zeros((imga.shape[0]+4, 407, 3))
    # img_right = np.zeros((imgb.shape[0]+4, imgb.shape[1]-12, 3))
    #
    # img_left[:imga.shape[0], :, :] = imga[:, :407, :]
    # img_right[4:, :, :] = imgb[:, 12:, :]
    # new_img = np.concatenate((img_left, img_right), axis=1)
    # cv2.imwrite("123456789.jpg", new_img)

    # -------- blending method --------
    img = cv2.imread("before blending.png")
    plt.imshow(img), plt.show()
    # x1 = 127
    # x2 = 234
    # x3 = 313
    # x4 = 397
    # x5 = 522
    # x6 = 593
    # x7 = 671
    # x8 = 796
    # record edge locations
    img1 = cv2.imread("landscape_1.jpg")
    img2 = cv2.imread("landscape_2.jpg")
    img3 = cv2.imread("landscape_3.jpg")
    img4 = cv2.imread("landscape_4.jpg")
    img5 = cv2.imread("landscape_5.jpg")
    img6 = cv2.imread("landscape_6.jpg")
    img7 = cv2.imread("landscape_7.jpg")
    img8 = cv2.imread("landscape_8.jpg")


