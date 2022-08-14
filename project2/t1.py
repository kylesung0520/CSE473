# Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions.
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Helper Functions Start

# https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
class match_components:
    def __init__(self, queryIndex, trainIndex, distance):
        self.queryIndex = queryIndex
        self.trainIndex = trainIndex
        self.distance = distance

def KNN_matcher(result, train_feature, query_feature, K):
    loop_range = train_feature.shape[0]
    for idx in range(loop_range):
        tmp_feature = train_feature[idx]
        # Euclidean distance to calculate distance for KNN matching
        distances = np.linalg.norm(tmp_feature - query_feature, axis=1, ord=2)
        # Find index of the K closest feature
        k_closest = distances.argsort(axis=0)[:K]
        # Save the queryIndex, trainIndex and distance to find homography
        result.append((match_components(idx, k_closest[0], distances[k_closest[0]]),
                       match_components(idx, k_closest[1], distances[k_closest[1]])))
    return result

# Helper Functions End

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    # Using SIFT to extract key points and features
    descriptor = cv2.SIFT_create()
    tmp_img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    tmp_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Extract a set of key points for each image
    # and extract features from each key point
    key_points1, features1 = descriptor.detectAndCompute(tmp_img1, None)
    key_points2, features2 = descriptor.detectAndCompute(tmp_img2, None)

    # Match features and use matches to determine if there is overlap between given pairs of images.
    feature1_result = KNN_matcher([], features1, features2, 2)
    feature2_result = KNN_matcher([], features2, features1, 2)

    cross_check = []
    for idx in range(len(feature1_result)):
        matches = feature1_result[idx]
        if (feature2_result[matches[0].trainIndex][0].trainIndex == idx) and (feature2_result[matches[1].trainIndex][1].trainIndex == idx):
            cross_check.append(matches)

    # ratio test to find good result
    ratio = 0.75
    good_matches = []
    for x in cross_check:
        if x[0].distance < x[1].distance * ratio: good_matches.append(x[0])

    # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    # Compute the homography between the overlapping pairs as needed. Use RANSAC to optimize your result
    tmp_key_points1 = np.float32([key_point.pt for key_point in key_points1])
    tmp_key_points2 = np.float32([key_point.pt for key_point in key_points2])
    if len(good_matches) > 4:
        points1 = np.float32([tmp_key_points1[x.queryIndex] for x in good_matches])
        points2 = np.float32([tmp_key_points2[x.trainIndex] for x in good_matches])
        (H, status) = cv2.findHomography(points1, points2, cv2.RANSAC, ransacReprojThreshold=4)
    else:
        return None

    M = (good_matches, H, status)
    (good_matches, H, status) = M

    # Transform the images and stitch the two images into one mosaic, eliminating the foreground
    # as described above, but do NOT crop your image.
    # Warping
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0] + img2.shape[0]))
    # Decided to set the img1 as foreground
    result[0:len(img2[0]), 0:img2.shape[1]] = img2

    cv2.imwrite(savepath, result)
    return

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
